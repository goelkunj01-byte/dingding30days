import discord
from discord.ext import commands, tasks
import asyncio
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone, time 
from discord import ui, ButtonStyle, Interaction, Embed
from discord import app_commands
import sys
import re 
import base64
from operator import itemgetter
from typing import Optional
import aiohttp 
from collections import defaultdict, OrderedDict
import io
import random
import hashlib
from urllib.parse import quote
from groq import Groq

# NEW: optional dependency for ?familytree diagram rendering. Wrapped in
# try/except so a missing package doesn't crash the whole bot -- ?familytree
# just reports unavailable until matplotlib+networkx are added to requirements.txt.
try:
    import matplotlib
    matplotlib.use("Agg")  # headless rendering -- no display available on a host like Railway
    import matplotlib.pyplot as plt
    import networkx as nx
    FAMILY_TREE_AVAILABLE = True
except ImportError:
    FAMILY_TREE_AVAILABLE = False
    print("⚠️ matplotlib/networkx not installed -- ?familytree will be unavailable until they're added to requirements.txt.")

# NEW: optional dependency for ?resizepfp image resizing (letterbox-to-square
# so nothing gets cropped when Discord forces the avatar into a square).
# Wrapped in try/except so a missing package doesn't crash the whole bot --
# ?resizepfp just reports unavailable until Pillow is added to requirements.txt.
try:
    from PIL import Image, ImageFilter
    PFP_TOOLS_AVAILABLE = True
except ImportError:
    PFP_TOOLS_AVAILABLE = False
    print("⚠️ Pillow not installed -- ?resizepfp will be unavailable until it's added to requirements.txt.")

# NEW: Load variables from a .env file sitting next to bot.py.
# Useful on hosts that don't expose an environment-variables UI.
load_dotenv()

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --------------------------------------------------------
# 🧠 AI INTEGRATIONS (Groq for most commands, Gemini only for ?rate)
# --------------------------------------------------------

# Groq for Fast Text (Riddles, Poems, Chat)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_TEXT_MODEL = "openai/gpt-oss-120b"  # used by everything except ?talk/?bnstory: ?vibecheck, ?roast, riddles, ?rate fallback text, etc.
# UPDATED (fix for Groq's 08/16/26 shutdown of llama-3.3-70b-versatile, take
# 2): ?talk/?bnstory were pinned to llama-3.3-70b-versatile for its
# personality/flirty-roleplay tone. moonshotai/kimi-k2-instruct-0905 was
# tried as a replacement, but Groq deprecated it back on 03/23/26 in favor of
# openai/gpt-oss-120b -- it 404s with model_not_found on current accounts.
# As of 06/17/26 Groq also retired llama-3.1-8b-instant, qwen/qwen3-32b, and
# meta-llama/llama-4-scout, funneling almost everything into openai/gpt-oss-
# 120b/20b. Of what's actually still live on Groq, qwen/qwen3.6-27b (same
# model GROQ_VISION_MODEL already uses) is the closest fit -- a real
# general-purpose chat model, and noticeably less locked-down for flirty/
# romantic roleplay than OpenAI's gpt-oss models. It IS a reasoning model, so
# it needs reasoning_effort suppressed like the others below (but NOT
# combined with reasoning_format, which is the specific combo that 400s --
# see the GROQ_VISION_REASONING_KWARGS note just below).
GROQ_CHAT_MODEL = "qwen/qwen3.6-27b"
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"  # Groq marks this a preview model -- could change/move without much notice, but it's the only vision option Groq currently offers

# BUG FIX: both GROQ_TEXT_MODEL and GROQ_VISION_MODEL are "reasoning" models
# (thinking mode). Without telling the API to suppress that, their internal
# chain-of-thought reasoning gets included in .content -- which is exactly
# what caused ?rate/auto-caption to blow way past Discord's length limits,
# and also why ?rate looked "random"/incoherent (you were seeing raw
# reasoning traces, not a clean final answer). include_reasoning=False and
# reasoning_effort="low"/"none" tell the model to skip/hide that and just
# return the final answer.
GROQ_TEXT_REASONING_KWARGS = {"include_reasoning": False, "reasoning_effort": "low"}
GROQ_VISION_REASONING_KWARGS = {"reasoning_effort": "none", "reasoning_format": "hidden"}  # BUG FIX: include_reasoning + reasoning_format together = 400 invalid_request_error, Groq rejects the combo
# NEW: qwen3.6-27b for chat -- deliberately just reasoning_effort, no
# include_reasoning/reasoning_format, to steer clear of the same 400 error
# noted above for the vision kwargs. IMPORTANT: unlike gpt-oss models,
# qwen3.6-27b only accepts reasoning_effort of "none" or "default" -- "low"
# 400s on it ({"message": "`reasoning_effort` must be one of `none` or
# `default`"}).
GROQ_CHAT_REASONING_KWARGS = {"reasoning_effort": "none"}
# BUG FIX: qwen/qwen3.6-27b's default max output length (~2048 tokens) blows
# straight through the free/on-demand tier's Output Tokens Per Minute (OTPM)
# limit of 1000 in a SINGLE request -- Groq rejects it outright with a 429
# ("Request too large... Requested 2048... Limit 1000"), so ?talk and
# ?bnstory failed on literally every message. Capping max_tokens well under
# that limit fixes it. Applies to every call using GROQ_CHAT_MODEL or
# GROQ_VISION_MODEL, since they're currently the same model and share the
# same per-organization OTPM quota.
GROQ_QWEN_MAX_TOKENS = 600

# Create the client and start a chat session
def get_groq_text(prompt):
    completion = groq_client.chat.completions.create(
        model=GROQ_TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        **GROQ_TEXT_REASONING_KWARGS,
    )
    return completion.choices[0].message.content

# NEW: Sends an image + prompt to Groq's vision model. Mirrors
# get_gemini_vision_text's signature so callers (?rate, auto-caption) can
# use either interchangeably. Uses a base64 data URI since that's simplest
# for images the bot already has in memory (avatars, uploaded attachments).
def get_groq_vision_text(image_bytes: bytes, mime_type: str, prompt: str, temperature: Optional[float] = None) -> str:
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    kwargs = dict(GROQ_VISION_REASONING_KWARGS)
    kwargs["max_tokens"] = GROQ_QWEN_MAX_TOKENS
    if temperature is not None:
        kwargs["temperature"] = temperature
    completion = groq_client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
            ],
        }],
        **kwargs,
    )
    return completion.choices[0].message.content

# NEW: Gemini client -- used for ?rate (text) AND birthday image generation.
# Wrapped in try/except so a missing package or key doesn't crash the whole bot;
# it just means those features report an error until GEMINI_API_KEY is set.
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"  # "Nano Banana" -- Gemini's native image generation model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None
genai_types = None
if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types as genai_types
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"⚠️ Gemini client failed to initialize (?rate/birthday images will be unavailable): {e}")
else:
    # BUG FIX: previously we called genai.Client(api_key=None) even when the key
    # was missing. The SDK doesn't fail immediately in that case -- it silently
    # falls back to Google's OAuth/ADC auth flow, which then fails at request
    # time with a confusing "ACCESS_TOKEN_TYPE_UNSUPPORTED" error instead of a
    # clear "no API key" message. Now we just skip creating the client entirely.
    print("⚠️ GEMINI_API_KEY is not set -- ?rate and birthday images will be unavailable until it's added.")

def get_gemini_text(prompt: str) -> str:
    if gemini_client is None:
        raise RuntimeError("Gemini client is not configured. Check that GEMINI_API_KEY is set correctly.")
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text

# NEW: Sends an actual image + prompt to Gemini so it can analyze what's really
# in the picture, instead of just guessing from a username (used by ?rate).
def get_gemini_vision_text(image_bytes: bytes, mime_type: str, prompt: str, temperature: Optional[float] = None) -> str:
    if gemini_client is None or genai_types is None:
        raise RuntimeError("Gemini client is not configured. Check that GEMINI_API_KEY is set correctly.")
    image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    kwargs = {}
    if temperature is not None:
        # NEW: default Gemini temperature tends to produce very "safe", samey
        # responses (e.g. always landing on 4/5). Bumping this for ?rate makes
        # scores actually spread out and reflect the specific image instead of
        # regressing to the same middle-ish answer every time.
        kwargs["config"] = genai_types.GenerateContentConfig(temperature=temperature)
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[image_part, prompt],
        **kwargs,
    )
    return response.text

# NEW: Generates a one-off birthday image using Gemini's native image model.
# Returns raw image bytes (PNG), or None if generation isn't available/fails --
# callers should fall back to a GIF in that case.
def generate_gemini_image(prompt: str) -> Optional[bytes]:
    if gemini_client is None or genai_types is None:
        return None
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        for part in response.parts:
            if part.inline_data:
                return part.inline_data.data
    except Exception as e:
        print(f"⚠️ Gemini image generation failed: {e}")
    return None

# NEW: Conversational version used by ?talk so the bot remembers context per-user.
# user_chats[user_id] stores a rolling list of {"role": ..., "content": ...} messages.
MAX_CHAT_HISTORY = 20  # number of messages (user+assistant combined) kept per user

def get_groq_chat_response(user_id: int, prompt: str) -> str:
    history = user_chats.get(user_id, [])
    messages = history + [{"role": "user", "content": prompt}]

    completion = groq_client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=messages,
        max_tokens=GROQ_QWEN_MAX_TOKENS,
        **GROQ_CHAT_REASONING_KWARGS,
    )
    reply = completion.choices[0].message.content

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": reply})
    user_chats[user_id] = history[-MAX_CHAT_HISTORY:]

    return reply

# Place this at the top of your script with your other global variables
user_chats = {}

# NEW: Lets people continue a ?talk conversation by just replying to the
# bot's message, instead of retyping "?talk" every time.
# Maps bot_message_id -> user_id so we know which reply belongs to which
# user's conversation. Capped so it doesn't grow forever.
active_talk_messages = OrderedDict()
MAX_TRACKED_TALK_MESSAGES = 500

def track_talk_message(message_id: int, user_id: int):
    active_talk_messages[message_id] = user_id
    if len(active_talk_messages) > MAX_TRACKED_TALK_MESSAGES:
        active_talk_messages.popitem(last=False)  # drop oldest

# --------------------------------------------------------
# --- BOT CONFIGURATION ---
# --------------------------------------------------------
AI_CHANNEL_ID = 1462463900451737745
# Replace with your bot's token 
MOD_LOG_CHANNEL_NAME = "mod-log"
MAX_MESSAGE_LENGTH = 2000
IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))

# --- DATA FILES ---
DATA_DIR = os.getenv("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)

def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)

AFK_FILE = data_path('afk.json')
WARNINGS_FILE = data_path('warnings.json')
MESSAGES_FILE = data_path('messages.json')
REMINDERS_FILE = data_path('reminders.json')
MEMBER_HISTORY_FILE = data_path('member_history.json')
BIRTHDAYS_FILE = data_path('birthdays.json')
SERVER_CONFIG_FILE = data_path('server_config.json')
MARRIAGES_FILE = data_path('marriages.json')
MARRIAGE_STATS_FILE = data_path('marriage_stats.json')
FRIENDS_FILE = data_path('friends.json')
EMOJI_SPAM_FILE = data_path('emoji_spam.json')
BIRTHDAY_ROLE_CLEANUP_FILE = data_path('birthday_role_cleanups.json')
TRIGGERS_FILE = data_path('triggers.json')
TRIGGER_IMAGES_DIR = data_path('trigger_images')
os.makedirs(TRIGGER_IMAGES_DIR, exist_ok=True)

TIMECAPSULES_FILE = data_path('timecapsules.json')
MEMORY_BANK_FILE = data_path('memory_bank.json')
TEMP_ROLES_FILE = data_path('temp_roles.json')
SERVER_MOOD_FILE = data_path('server_mood.json')
PRIVATE_CHANNELS_FILE = data_path('private_channels.json')  # NEW: { channel_id: {guild_id, owner_id, view: [user_ids], type: [user_ids]} } -- member-owned private channels (?createpvtchannel)
CUSTOM_AVATARS_FILE = data_path('custom_avatars.json')  # NEW: { user_id: {"path": ..., "set_by": ..., "set_at": ...} } -- GLOBAL (not per-server), so a mod-set ?av override follows the user everywhere this bot is
CUSTOM_AVATARS_DIR = data_path('custom_avatars')
os.makedirs(CUSTOM_AVATARS_DIR, exist_ok=True)
CONFESSIONS_FILE = data_path('confessions.json')  # NEW: { guild_id: count } -- running anonymous-confession counter per server
DIGEST_OPTOUT_FILE = data_path('digest_optout.json')  # NEW: [user_id, ...] -- users who opted out of the weekly personalized digest DM
VOICE_TIME_FILE = data_path('voice_time_together.json')  # NEW: { guild_id: { "id1-id2" (sorted pair): total_seconds_together } } -- for ?twin
RATE_USAGE_FILE = data_path('rate_usage.json')  # NEW: { user_id: iso_timestamp_of_last_use } -- enforces the 1/day ?rate limit
ROLE_MENU_CONFIG_FILE = data_path('role_menu_config.json')  # NEW: { guild_id: { category_name: [role_name, ...] } } -- ?setup_roles categories/options, mod-extendable via ?addnewcategory/?addnewsetup
STORY_SESSIONS_FILE = data_path('story_sessions.json')  # NEW: { thread_id: {...} } -- active ?bnstory roleplay sessions, see the ?bnstory section below
SCHEDULED_CONTROL_ACTIONS_FILE = data_path('scheduled_control_actions.json')  # NEW: { action_id: {...} } -- delayed/auto-revert actions from the natural-language control channel, see that section below

# --- LEADERBOARD CONFIG ---
LEADERBOARD_CHANNEL_NAME = "general" 
LEADERBOARD_CHANNEL_ID = os.getenv("LEADERBOARD_CHANNEL_ID", "1455502594947551254")
WINNER_ROLE_NAME = "The Chatterbox" 

# --- BIRTHDAY CONFIG ---
BIRTHDAY_CHANNEL_NAME = "general"
BIRTHDAY_CHANNEL_ID = os.getenv("BIRTHDAY_CHANNEL_ID", "1455502594947551254")
BIRTHDAY_ROLE_DURATION_SECONDS = 60 * 60 * 24
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")

STATUS_COMMAND_OWNERS = {"kanjuubarfiiii", "huh.ashh"}

KALA_MAJDUR_IMAGE_PATH = "kala_majdur.jpg"

# NEW: emoji list used by ?poll to react with number emojis.
NUMBER_EMOJIS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

current_bot_status = discord.Status.online
current_bot_activity = None
slash_commands_synced = False  # NEW: guards against re-syncing the slash command tree on every reconnect

# --- INTENTS & BOT INITIALIZATION ---
intents = discord.Intents.default()
intents.members = True 
intents.message_content = True 
intents.voice_states = True  # NEW: needed to track VC time together for ?twin

bot = commands.Bot(command_prefix='?', intents=intents, help_command=None)
bot.http_session = None
    
spam_tracker = defaultdict(lambda: {"messages": [], "strikes": 0, "last_strike_time": None})

rep_cooldowns = {}

# --- 2. FILE & DATA INITIALIZATION ---

HIGHLIGHTS_FILE = data_path('highlights.json')
RESTRICTED_WORDS_FILE = data_path('restricted_words.json')
REPUTATION_FILE = data_path('reputation.json')

def load_data(filename, default={}):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {filename} is corrupted. Starting with default data.")
                return default
    return default

def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

highlights = load_data(HIGHLIGHTS_FILE, default={})
restricted_words = load_data(RESTRICTED_WORDS_FILE, default=[])
if restricted_words is None:
    restricted_words = []

reputation = load_data(REPUTATION_FILE, default={})
if reputation is None:
    reputation = {}

hint_tracker = {}
user_chats = {}

afk_users = load_data(AFK_FILE)
warnings_data = load_data(WARNINGS_FILE)
message_counts = load_data(MESSAGES_FILE, default={})
if message_counts is None:
    message_counts = {}
reminders_data = load_data(REMINDERS_FILE)
member_history = load_data(MEMBER_HISTORY_FILE)
birthdays = load_data(BIRTHDAYS_FILE)
server_config = load_data(SERVER_CONFIG_FILE, default={})
# BUG FIX: marriages/marriage_stats/friends used to be flat { user_id: ... }
# dicts with no server dimension at all -- so ?marriages, ?couple, ?friendslist
# etc showed data combined across every server the bot is in. Now nested as
# { guild_id: { user_id: ... } } so each server's relationships are independent.
# NOTE: this is a breaking format change -- any marriages/friends recorded
# before this update lived in the old flat format with no guild attached, so
# they can't be automatically attributed to a specific server and won't carry
# forward. Existing users will need to ?marry / ?friend again post-update.
marriages = load_data(MARRIAGES_FILE, default={})
marriage_stats = load_data(MARRIAGE_STATS_FILE, default={})
friends = load_data(FRIENDS_FILE, default={})

def get_guild_marriages(guild_id) -> dict:
    return marriages.setdefault(str(guild_id), {})

def get_guild_marriage_stats(guild_id) -> dict:
    return marriage_stats.setdefault(str(guild_id), {})

def get_guild_friends(guild_id) -> dict:
    return friends.setdefault(str(guild_id), {})

emoji_spam_targets = load_data(EMOJI_SPAM_FILE, default={})
if emoji_spam_targets is None:
    emoji_spam_targets = {}
birthday_role_cleanups = load_data(BIRTHDAY_ROLE_CLEANUP_FILE, default={})
if birthday_role_cleanups is None:
    birthday_role_cleanups = {}
triggers = load_data(TRIGGERS_FILE, default={})
if triggers is None:
    triggers = {}
timecapsules = load_data(TIMECAPSULES_FILE, default={})
if timecapsules is None:
    timecapsules = {}
memory_bank = load_data(MEMORY_BANK_FILE, default={})
if memory_bank is None:
    memory_bank = {}
temp_roles = load_data(TEMP_ROLES_FILE, default={})
if temp_roles is None:
    temp_roles = {}
server_mood = load_data(SERVER_MOOD_FILE, default={})
if server_mood is None:
    server_mood = {}
private_channels = load_data(PRIVATE_CHANNELS_FILE, default={})  # NEW: { channel_id: {guild_id, owner_id, view: [...], type: [...]} }
if private_channels is None:
    private_channels = {}
custom_avatars = load_data(CUSTOM_AVATARS_FILE, default={})  # NEW: { user_id: {"path": ..., "set_by": ..., "set_at": ...} } -- global override for ?av
if custom_avatars is None:
    custom_avatars = {}
confessions_data = load_data(CONFESSIONS_FILE, default={})  # NEW: { guild_id: count }
if confessions_data is None:
    confessions_data = {}
digest_optout = load_data(DIGEST_OPTOUT_FILE, default=[])  # NEW: [user_id, ...]
if digest_optout is None:
    digest_optout = []
voice_time_together = load_data(VOICE_TIME_FILE, default={})  # NEW: { guild_id: { "id1-id2": seconds } }
if voice_time_together is None:
    voice_time_together = {}
rate_usage = load_data(RATE_USAGE_FILE, default={})  # NEW: { user_id: iso_timestamp }
if rate_usage is None:
    rate_usage = {}
role_menu_config = load_data(ROLE_MENU_CONFIG_FILE, default={})  # NEW: { guild_id: { category_name: [role_name, ...] } }
if role_menu_config is None:
    role_menu_config = {}
story_sessions = load_data(STORY_SESSIONS_FILE, default={})  # NEW: { thread_id: {...} } -- see ?bnstory section
if story_sessions is None:
    story_sessions = {}
scheduled_control_actions = load_data(SCHEDULED_CONTROL_ACTIONS_FILE, default={})  # NEW: { action_id: {...} } -- see the control-channel section
if scheduled_control_actions is None:
    scheduled_control_actions = {}
# In-memory only (doesn't need to survive a restart -- worst case one VC
# session's time before a redeploy doesn't get counted, no big deal):
# { (guild_id, channel_id): { user_id: join_datetime } }
active_voice_sessions = {}

# --- LOGGING UTILITIES ---

async def delete_thread_later(thread,delay):
    await asyncio.sleep(delay)
    try:
        await thread.delete()
    except:
        pass

# NEW: explicitly blocks one named account from trigger-related and emoji-spam
# commands, regardless of whether they hold Manage Messages -- a deliberate
# override on top of normal permissions, per server owner's request.
# NOTE: matches on Discord username (the @handle, not server nickname) since
# that's what was given -- if this account ever changes its username, this
# block needs updating to match the new one (or better, swap to matching by
# user ID if you have it, which never changes).
BLOCKED_USERNAMES_FOR_TRIGGERS_AND_SPAMEM = {"b1uelays"}

async def not_blocked_from_trigger_commands(ctx) -> bool:
    if ctx.author.name.lower() in BLOCKED_USERNAMES_FOR_TRIGGERS_AND_SPAMEM:
        await ctx.send("❌ You're not permitted to use this command.")
        return False
    return True

def get_mod_log_channel(guild: discord.Guild):
    # UPDATED: checks this guild's configured mod-log channel (?setchannel modlog #channel)
    # first, then falls back to name-matching a channel called "mod-log".
    guild_settings = server_config.get(str(guild.id), {})
    channel_id = guild_settings.get("mod_log_channel_id")
    if channel_id:
        channel = guild.get_channel(int(channel_id))
        if channel:
            return channel
    return discord.utils.get(guild.text_channels, name=MOD_LOG_CHANNEL_NAME)

def find_channel(guild: discord.Guild, channel_id: Optional[str], *fallback_names: str):
    if channel_id:
        try:
            channel = guild.get_channel(int(channel_id))
            if channel:
                return channel
        except (ValueError, TypeError):
            pass
    for name in fallback_names:
        channel = discord.utils.get(guild.text_channels, name=name)
        if channel:
            return channel
    return None

def get_configured_channel(guild: discord.Guild, config_key: str, hardcoded_default_id: Optional[str] = None, *fallback_names: str):
    guild_settings = server_config.get(str(guild.id), {})
    channel_id = guild_settings.get(config_key) or hardcoded_default_id
    return find_channel(guild, channel_id, *fallback_names)

# NEW: ?talk channel restriction (?restricttalk / ?unrestricttalk). Same
# whitelist pattern as hall-of-fame sources and trigger channels: if a
# server has whitelisted any channels, ?talk only works there; if none are
# whitelisted, it works everywhere (backward compatible default).
def is_talk_allowed_in_channel(guild: discord.Guild, channel_id: int) -> bool:
    allowed = server_config.get(str(guild.id), {}).get("talk_allowed_channel_ids")
    if not allowed:
        return True
    result = str(channel_id) in allowed
    if not result:
        print(f"⚠️ ?talk blocked in channel {channel_id} (guild {guild.id}) -- allowed list: {allowed}")
    return result

# NEW: spam protection on/off toggle, server-wide or per-channel, set by the
# natural-language control channel (see that section below). Uses a lazily-
# checked expiry timestamp instead of a background task to auto-revert --
# once the stored expiry is in the past, protection is back on automatically
# the next time this is checked, no scheduled job needed for that part.
# Stored value is either `True` (disabled indefinitely) or an ISO timestamp
# string (disabled until that time).
def _spam_disable_value_active(value) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value) > datetime.now(timezone.utc)
        except Exception:
            return False
    return False

def is_spam_protection_disabled(guild_id: int, channel_id: int) -> bool:
    cfg = server_config.get(str(guild_id), {})
    if _spam_disable_value_active(cfg.get("spam_protection_disabled_until")):
        return True
    if _spam_disable_value_active(cfg.get("spam_protection_disabled_channels", {}).get(str(channel_id))):
        return True
    return False

# --------------------------------------------------------
# 🎭 SERVER MOOD
# --------------------------------------------------------
MOOD_LABELS = [
    (0.5, "unhinged 🌀"),
    (0.2, "hyped ⚡"),
    (-0.2, "chill 😌"),
    (-0.5, "quiet 🌙"),
    (float("-inf"), "dead 💀"),
]

def mood_label_for_score(score: float) -> str:
    for threshold, label in MOOD_LABELS:
        if score >= threshold:
            return label
    return "chill 😌"

def update_server_mood(guild_id: int, message: discord.Message):
    gid = str(guild_id)
    state = server_mood.get(gid, {"score": 0.0, "last_updated": datetime.now(timezone.utc).isoformat()})

    last_updated = datetime.fromisoformat(state["last_updated"])
    minutes_elapsed = (datetime.now(timezone.utc) - last_updated).total_seconds() / 60
    decay = min(1.0, minutes_elapsed * 0.05)
    score = state["score"] * (1 - decay)

    content = message.content
    if len(content) > 5 and content.isupper():
        score += 0.06
    if content.count("!") >= 2:
        score += 0.04
    if len(message.mentions) >= 3:
        score += 0.05
    if not content.strip():
        score -= 0.01

    score = max(-1.0, min(1.0, score))
    server_mood[gid] = {"score": score, "last_updated": datetime.now(timezone.utc).isoformat()}



async def send_mod_log(guild, title, description, moderator: discord.User):
    channel = get_mod_log_channel(guild)
    if channel:
        embed = discord.Embed(title=title, description=description, color=discord.Color.orange())
        embed.set_footer(text=f"Moderator: {moderator.name}#{moderator.discriminator}", 
                             icon_url=moderator.display_avatar.url)
        embed.timestamp = datetime.now(timezone.utc)
        await channel.send(embed=embed)

def split_message_chunks(text: str) -> list[str]:
    if len(text) <= MAX_MESSAGE_LENGTH: return [text]
    
    chunks = []
    current_chunk = ""
    for line in text.split('\n'):
        if len(current_chunk) + len(line) + 1 > MAX_MESSAGE_LENGTH:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += '\n' + line if current_chunk else line
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# NEW: defensive safety net for single-shot AI outputs that go straight into
# an embed description or a plain message (rather than through safe_send's
# chunking) -- e.g. ?rate, auto-caption. Even with reasoning suppressed at
# the API level, this guarantees a stray oversized response can never crash
# the command with a Discord 400 (Invalid Form Body) again.
def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 1].rstrip() + "…"

async def safe_send(destination, text: str, reply_to=None):
    chunks = split_message_chunks(text)
    sent_messages = []
    for i, chunk in enumerate(chunks):
        if i == 0 and reply_to is not None:
            msg = await reply_to.reply(chunk)
        else:
            msg = await destination.send(chunk)
        sent_messages.append(msg)
    return sent_messages


# --------------------------------------------------------
# ⚙️ BACKGROUND TASKS AND UTILITIES
# --------------------------------------------------------

def parse_reminder_time(date_str: str, time_str: str) -> datetime:
    if ':' not in time_str: time_str += ":00"
    full_str = f"{date_str} {time_str}"
    try:
        naive_dt = datetime.strptime(full_str, '%d/%m/%Y %H:%M') 
    except ValueError:
        raise ValueError("Invalid date or time format. Please use **DD/MM/YYYY HH:MM** (e.g., 25/12/2026 14:30)")
    
    localized_dt_ist = naive_dt.replace(tzinfo=IST_TIMEZONE)
    localized_dt_utc = localized_dt_ist.astimezone(timezone.utc)
    if localized_dt_utc <= datetime.now(timezone.utc):
        raise ValueError("The reminder time must be in the future.")
    return localized_dt_utc

async def create_reminder(ctx, title: str, date_str: str, time_str: str, private: bool, recipient_id: int):
    try:
        reminder_time_utc = parse_reminder_time(date_str, time_str)
    except ValueError as e:
        return await ctx.send(f"❌ Time Error: {e}")

    reminder_id = str(datetime.now().timestamp())
    reminders_data[reminder_id] = {
        'user_id': str(ctx.author.id),
        'channel_id': str(ctx.channel.id),
        'title': title,
        'time_utc': reminder_time_utc.isoformat(),
        'recipient_id': str(recipient_id)
    }
    save_data(reminders_data, REMINDERS_FILE)

    confirmation_time_ist = reminder_time_utc.astimezone(IST_TIMEZONE).strftime('%A, %d %B at %I:%M %p IST')
    
    is_self_reminder = recipient_id == ctx.author.id
    
    embed_title = "✅ Reminder Set!" if is_self_reminder else "✅ DM Scheduled!"
    
    embed = discord.Embed(
        title=embed_title,
        description=f"**Title:** {title}\n**Delivery:** {confirmation_time_ist}",
        color=discord.Color.green()
  )
    
    try:
        recipient = await bot.fetch_user(recipient_id)
        if not is_self_reminder:
             embed.add_field(name="Recipient", value=recipient.mention, inline=False)
        embed.set_footer(text=f"The message will be sent to {recipient.display_name}'s DM.")
    except discord.NotFound:
        if not is_self_reminder:
             embed.add_field(name="Recipient ID", value=recipient_id, inline=False)
        embed.set_footer(text=f"The message will be sent to user ID {recipient_id}'s DM.")
    
    if private:
        try:
            await ctx.message.delete()
        except discord.Forbidden:
            print("Warning: Bot lacks permissions to delete command messages.")
            
        await ctx.reply(embed=embed, ephemeral=True)
    else:
        await ctx.send(f"🔔 Reminder set by {ctx.author.mention}!", embed=embed)
        
async def remove_winner_role_after_delay(guild_id, member_id, role_id, delay_seconds, channel_id):
    await asyncio.sleep(delay_seconds)
    guild = bot.get_guild(guild_id)
    member = guild.get_member(member_id) if guild else None
    role = guild.get_role(role_id) if guild else None
    if member and role in member.roles:
        try:
            await member.remove_roles(role, reason="Weekly Chatterbox Role Expiration.")
            channel = bot.get_channel(channel_id)
            if channel:
                await channel.send(f"👑 {member.mention}'s **{role.name}** role has expired. Congrats on your win last week!")
        except discord.Forbidden:
            print(f"Error removing role {role.name} from {member.name}: Forbidden.")
        except Exception as e:
            print(f"Error during role removal: {e}")

# --------------------------------------------------------
# 🎂 BIRTHDAY SYSTEM
# --------------------------------------------------------

async def generate_pollinations_image(prompt: str) -> Optional[bytes]:
    session = bot.http_session
    if not session:
        print("⚠️ Pollinations skipped: bot.http_session is not initialized yet.")
        return None
    try:
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        async with session.get(
            url,
            params={"width": "1024", "height": "1024", "nologo": "true"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                data = await resp.read()
                if data:
                    return data
                print("⚠️ Pollinations returned 200 but no image data.")
            else:
                body_preview = (await resp.text())[:200]
                print(f"⚠️ Pollinations returned status {resp.status}: {body_preview}")
    except Exception as e:
        print(f"⚠️ Pollinations image generation failed: {e}")
    return None

async def get_birthday_image_bytes(prompt: str) -> Optional[bytes]:
    return await generate_pollinations_image(prompt)

async def get_birthday_gif() -> Optional[str]:
    session = bot.http_session
    if GIPHY_API_KEY and session:
        try:
            async with session.get(
                "https://api.giphy.com/v1/gifs/random",
                params={"api_key": GIPHY_API_KEY, "tag": "happy birthday cake balloons", "rating": "g"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    url = data.get("data", {}).get("images", {}).get("original", {}).get("url")
                    if url:
                        return url
                else:
                    print(f"⚠️ Giphy returned status {resp.status}")
        except Exception as e:
            print(f"⚠️ Giphy fetch failed: {e}")
    return None

# --------------------------------------------------------
# 🇮🇳 REUSABLE ENGLISH <-> HINGLISH TOGGLE
# --------------------------------------------------------
class LanguageToggleView(ui.View):
    def __init__(self, base_prompt: str, embed_builder, current_lang: str = "en"):
        super().__init__(timeout=180)
        self.base_prompt = base_prompt
        self.embed_builder = embed_builder
        self.current_lang = current_lang
        self.message = None
        self._set_button_label()

    def _set_button_label(self):
        self.toggle_button.label = "🇬🇧 Switch to English" if self.current_lang == "hi" else "🇮🇳 Switch to Hinglish"

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass

    @ui.button(label="🇮🇳 Switch to Hinglish", style=discord.ButtonStyle.secondary)
    async def toggle_button(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.defer()

        target_lang = "English" if self.current_lang == "hi" else "Hinglish"
        lang_instruction = (
            "\n\nWrite this entirely in natural, casual Hinglish (the way people actually type it in "
            "Indian Discord/WhatsApp chats -- Hindi and English mixed together, informal, not formal "
            "textbook Hindi)." if target_lang == "Hinglish"
            else "\n\nWrite this entirely in natural English."
        )

        try:
            new_text = await asyncio.to_thread(get_groq_text, self.base_prompt + lang_instruction)
        except Exception as e:
            print(f"⚠️ Language toggle failed: {e}")
            return await interaction.followup.send("❌ Couldn't switch language right now, try again in a bit.", ephemeral=True)

        self.current_lang = "hi" if target_lang == "Hinglish" else "en"
        self._set_button_label()
        new_embed = self.embed_builder(new_text.strip())
        try:
            await interaction.message.edit(embed=new_embed, view=self)
        except Exception as e:
            print(f"⚠️ Failed to edit message after language toggle: {e}")

def _birthday_context(member: discord.Member, guild: discord.Guild) -> str:
    gid = str(guild.id)
    uid = str(member.id)
    counts = message_counts.get(gid, {})
    msg_count = counts.get(uid, 0)

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    rank = next((i + 1 for i, (u, c) in enumerate(sorted_counts) if u == uid), None)

    role_names = [r.name for r in member.roles if r.name != "@everyone"]

    record = member_history.get(gid, {}).get(uid, {})
    join_count = record.get("join_count", 1)

    return (
        f"Display name: {member.display_name}\n"
        f"Roles: {', '.join(role_names) if role_names else 'no special roles'}\n"
        f"Messages sent this week: {msg_count}\n"
        f"Chat activity rank in server: {rank if rank else 'not currently ranked'}\n"
        f"Times they've joined this server: {join_count}\n"
    )

async def generate_birthday_message(member: discord.Member, guild: discord.Guild, dm: bool) -> str:
    context = _birthday_context(member, guild)

    if dm:
        prompt = (
            "Write a warm, personal, slightly playful happy birthday DM (4-6 sentences) for a Discord "
            f"user, based on this context about them:\n{context}\n"
            "Make them feel genuinely special and seen -- weave in their vibe/activity naturally "
            "instead of just listing stats back at them. No generic corporate language. "
            "Sprinkle in 4-6 relevant emojis naturally throughout the message (not just at the start/end). "
            "Sign off warmly. Output only the message, nothing else."
        )
    else:
        prompt = (
            "Write a short, punchy, fun public happy-birthday shoutout (2-4 sentences) "
            f"for a Discord server to celebrate this user, based on:\n{context}\n"
            "Make it feel unique to their personality/activity in the server, not generic. "
            "Mention their name naturally. Sprinkle in 4-6 relevant emojis naturally throughout "
            "the message (not just at the start/end). Output only the message, nothing else."
        )

    try:
        return await asyncio.to_thread(get_groq_text, prompt)
    except Exception as e:
        print(f"⚠️ Birthday message generation failed, using fallback: {e}")
        return f"🎉 Happy Birthday, {member.mention}! Hope your day is absolutely amazing! 🎂"

async def generate_birthday_role_name(member: discord.Member, guild: discord.Guild) -> str:
    context = _birthday_context(member, guild)
    prompt = (
        "In 2 to 4 words, invent a fun, affectionate nickname/title for a Discord user based on this "
        f"context about their activity/vibe:\n{context}\n"
        "Something like 'The Chaotic Gremlin' or 'The Meme Lord' or 'The Quiet Legend'. "
        "Reply with ONLY the phrase itself -- no quotes, no punctuation, no explanation."
    )
    try:
        phrase = await asyncio.to_thread(get_groq_text, prompt)
        phrase = phrase.strip().strip('"').strip("'").strip()
    except Exception as e:
        print(f"⚠️ Birthday role name generation failed, using fallback: {e}")
        phrase = "The Birthday Legend"

    role_name = f"🎂 {phrase} — {member.display_name}"
    return role_name[:100]

async def perform_birthday_role_cleanup(guild_id: int, member_id: int, role_id: int):
    guild = bot.get_guild(guild_id)
    if not guild:
        return
    role = guild.get_role(role_id)
    member = guild.get_member(member_id)
    try:
        if role and member and role in member.roles:
            await member.remove_roles(role, reason="Birthday role expired (24h).")
        if role:
            await role.delete(reason="Birthday role expired (24h) -- cleaning up one-off role.")
    except discord.Forbidden:
        print(f"⚠️ Missing permissions to clean up birthday role {role_id} in guild {guild_id}.")
    except Exception as e:
        print(f"⚠️ Error cleaning up birthday role: {e}")
    finally:
        birthday_role_cleanups.pop(str(role_id), None)
        save_data(birthday_role_cleanups, BIRTHDAY_ROLE_CLEANUP_FILE)

async def cleanup_birthday_role_after_delay(guild_id: int, member_id: int, role_id: int, delay_seconds: int):
    await asyncio.sleep(delay_seconds)
    await perform_birthday_role_cleanup(guild_id, member_id, role_id)

# --------------------------------------------------------
# ⏳ GENERIC TEMP-ROLE SYSTEM
# --------------------------------------------------------

async def assign_temp_role(guild: discord.Guild, member: discord.Member, role_name: str,
                            duration_seconds: int, color: discord.Color = None,
                            reason: str = "Temporary role", delete_role: bool = True) -> discord.Role | None:
    try:
        role = await guild.create_role(name=role_name[:100], color=color or discord.Color.default(), reason=reason)
        await member.add_roles(role, reason=reason)
    except discord.Forbidden:
        print(f"⚠️ Missing permissions to create/assign temp role in {guild.name}.")
        return None
    except Exception as e:
        print(f"⚠️ Temp role creation failed: {e}")
        return None

    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)).isoformat()
    temp_roles[str(role.id)] = {
        "guild_id": guild.id,
        "member_id": member.id,
        "expires_at": expires_at,
        "delete_role": delete_role
    }
    save_data(temp_roles, TEMP_ROLES_FILE)
    bot.loop.create_task(cleanup_temp_role_after_delay(guild.id, member.id, role.id, duration_seconds, delete_role))
    return role

async def perform_temp_role_cleanup(guild_id: int, member_id: int, role_id: int, delete_role: bool = True):
    guild = bot.get_guild(guild_id)
    if not guild:
        return
    role = guild.get_role(role_id)
    member = guild.get_member(member_id)
    try:
        if role and member and role in member.roles:
            await member.remove_roles(role, reason="Temporary role expired.")
        if role and delete_role:
            await role.delete(reason="Temporary role expired -- cleaning up.")
    except discord.Forbidden:
        print(f"⚠️ Missing permissions to clean up temp role {role_id} in guild {guild_id}.")
    except Exception as e:
        print(f"⚠️ Error cleaning up temp role: {e}")
    finally:
        temp_roles.pop(str(role_id), None)
        save_data(temp_roles, TEMP_ROLES_FILE)

async def cleanup_temp_role_after_delay(guild_id: int, member_id: int, role_id: int, delay_seconds: int, delete_role: bool = True):
    await asyncio.sleep(delay_seconds)
    await perform_temp_role_cleanup(guild_id, member_id, role_id, delete_role)

async def celebrate_birthday(guild: discord.Guild, member: discord.Member, bday: dict) -> list[str]:
    issues = []
    if member.bot:
        return issues

    age = None
    if bday.get("year"):
        today = datetime.now(IST_TIMEZONE)
        age = today.year - bday["year"] - ((today.month, today.day) < (bday["month"], bday["day"]))

    channel = get_configured_channel(guild, "birthday_channel_id", BIRTHDAY_CHANNEL_ID, BIRTHDAY_CHANNEL_NAME, LEADERBOARD_CHANNEL_NAME)

    if not channel:
        issues.append(
            f"⚠️ Couldn't find the birthday announcement channel. A mod needs to run "
            f"`?setchannel birthday #channel-name` once to configure it for this server."
        )

    public_message = await generate_birthday_message(member, guild, dm=False)

    embed = discord.Embed(
        title="🎉🎂 HAPPY BIRTHDAY! 🎂🎉",
        description=public_message,
        color=discord.Color.random(),
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    if age:
        embed.add_field(name="🎈 Turning", value=str(age), inline=True)

    image_bytes = await get_birthday_image_bytes(
        f"A vibrant, colorful birthday celebration illustration with balloons, confetti, and a "
        f"birthday cake, festive party atmosphere, digital art style. No readable text in the image."
    )
    image_file = None
    if image_bytes:
        image_file = discord.File(io.BytesIO(image_bytes), filename="birthday.png")
        embed.set_image(url="attachment://birthday.png")
    else:
        gif_url = await get_birthday_gif()
        if gif_url:
            embed.set_image(url=gif_url)
        else:
            issues.append("⚠️ Couldn't generate or fetch a birthday image/GIF -- sent without one. Check the console logs for the exact reason (Pollinations/Hugging Face/Giphy).")

    if channel:
        try:
            send_kwargs = {"embed": embed, "allowed_mentions": discord.AllowedMentions(everyone=True)}
            if image_file:
                send_kwargs["file"] = image_file
            await channel.send(f"🎉🎊 **@everyone** 🎊🎉 It's {member.mention}'s birthday today! 🥳🎂", **send_kwargs)
        except discord.Forbidden:
            issues.append(f"⚠️ Missing permissions to send messages in #{channel.name} -- couldn't post the announcement.")
        except Exception as e:
            issues.append(f"⚠️ Error posting birthday announcement: {e}")

    role = None
    try:
        role_name = await generate_birthday_role_name(member, guild)
        role = await guild.create_role(name=role_name, color=discord.Color.random(), reason="Birthday celebration role")
        await member.add_roles(role, reason="Happy Birthday!")
    except discord.Forbidden:
        issues.append("⚠️ Missing **Manage Roles** permission -- couldn't create/assign the birthday role. Give the bot Manage Roles and make sure its role is positioned above where new roles get created.")
    except Exception as e:
        issues.append(f"⚠️ Birthday role error: {e}")

    if role:
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=BIRTHDAY_ROLE_DURATION_SECONDS)).isoformat()
        birthday_role_cleanups[str(role.id)] = {
            "guild_id": guild.id,
            "member_id": member.id,
            "expires_at": expires_at
        }
        save_data(birthday_role_cleanups, BIRTHDAY_ROLE_CLEANUP_FILE)

        bot.loop.create_task(
            cleanup_birthday_role_after_delay(guild.id, member.id, role.id, BIRTHDAY_ROLE_DURATION_SECONDS)
        )

    try:
        dm_message = await generate_birthday_message(member, guild, dm=True)
        dm_embed = discord.Embed(
            title="🎂✨ Happy Birthday! ✨🎂",
            description=dm_message,
            color=discord.Color.gold()
        )

        dm_image_bytes = await get_birthday_image_bytes(
            f"A warm, cheerful birthday illustration with balloons and a decorated cake, "
            f"personal and celebratory, digital art style. No readable text in the image."
        )
        dm_file = None
        if dm_image_bytes:
            dm_file = discord.File(io.BytesIO(dm_image_bytes), filename="birthday_dm.png")
            dm_embed.set_image(url="attachment://birthday_dm.png")
        else:
            dm_gif_url = await get_birthday_gif()
            if dm_gif_url:
                dm_embed.set_image(url=dm_gif_url)

        if dm_file:
            await member.send(embed=dm_embed, file=dm_file)
        else:
            await member.send(embed=dm_embed)
    except discord.Forbidden:
        issues.append(f"⚠️ Could not DM {member.display_name} for their birthday (their DMs are likely closed).")
    except Exception as e:
        issues.append(f"⚠️ Birthday DM error: {e}")

    return issues

# --------------------------------------------------------
# 🕰️ BACKGROUND TASK: REMINDER CHECK
# --------------------------------------------------------
@tasks.loop(minutes=1.0)
async def reminder_check_loop():
    global reminders_data
    if not reminders_data: return
    now_utc = datetime.now(timezone.utc)
    reminders_to_remove = []

    for reminder_id, data in reminders_data.items():
        try:
            time_utc_str = data.get('time_utc')
            if not time_utc_str:
                print(f"⚠️ Reminder {reminder_id} is missing 'time_utc' -- removing malformed entry.")
                reminders_to_remove.append(reminder_id)
                continue

            reminder_time = datetime.fromisoformat(time_utc_str)
            if reminder_time <= now_utc:
                
                recipient = await bot.fetch_user(int(data['recipient_id']))
                
                sender = bot.get_user(int(data['user_id']))
                sender_name = sender.display_name if sender else "A previous user"
                
                reminder_embed = discord.Embed(
                    title="🔔 Scheduled Message/Reminder!",
                    description=f"**{data['title']}**",
                    color=discord.Color.red()
                )
                reminder_embed.add_field(
                    name="Scheduled Time", 
                    value=f"{reminder_time.astimezone(IST_TIMEZONE).strftime('%A, %d %B %Y at %I:%M %p IST')}"
                )
                reminder_embed.set_footer(text=f"Sent by {sender_name}")
                
                try:
                    await recipient.send(embed=reminder_embed)
                except discord.Forbidden:
                    channel = bot.get_channel(int(data['channel_id']))
                    if channel:
                        await channel.send(f"⚠️ **DM Failed for {recipient.mention}:** The scheduled message titled '{data['title']}' could not be delivered because their DMs are likely disabled.")
                
                reminders_to_remove.append(reminder_id)
        except discord.NotFound:
            print(f"Recipient for reminder ID {reminder_id} not found. Removing reminder.")
            reminders_to_remove.append(reminder_id)
        except Exception as e:
            print(f"Error processing reminder ID {reminder_id}: {e}")
            reminders_to_remove.append(reminder_id) 

    for r_id in reminders_to_remove:
        if r_id in reminders_data: del reminders_data[r_id]
            
    if reminders_to_remove: save_data(reminders_data, REMINDERS_FILE)

@reminder_check_loop.error
async def reminder_check_loop_error(error):
    print(f"🚨 Reminder Check Loop Error: {error}")

# --------------------------------------------------------
# 📦 BACKGROUND TASK: TIME CAPSULE DELIVERY
# --------------------------------------------------------
@tasks.loop(minutes=1.0)
async def timecapsule_check_loop():
    if not timecapsules:
        return
    now_utc = datetime.now(timezone.utc)
    to_remove = []

    for capsule_id, data in timecapsules.items():
        try:
            deliver_at = datetime.fromisoformat(data['deliver_at'])
            if deliver_at <= now_utc:
                channel = bot.get_channel(data['channel_id'])
                author = bot.get_user(data['author_id'])
                author_name = author.display_name if author else "Someone"

                if channel:
                    embed = discord.Embed(
                        title="📦 A Time Capsule Has Opened!",
                        description=data['message'],
                        color=discord.Color.dark_gold(),
                        timestamp=now_utc
                    )
                    embed.set_footer(text=f"Sealed by {author_name}")
                    try:
                        await channel.send(f"📦 <@{data['author_id']}>'s time capsule just opened!", embed=embed)
                    except Exception as e:
                        print(f"⚠️ Failed to deliver time capsule {capsule_id}: {e}")
                to_remove.append(capsule_id)
        except Exception as e:
            print(f"⚠️ Error processing time capsule {capsule_id}: {e}")
            to_remove.append(capsule_id)

    for cid in to_remove:
        timecapsules.pop(cid, None)
    if to_remove:
        save_data(timecapsules, TIMECAPSULES_FILE)

@timecapsule_check_loop.error
async def timecapsule_check_loop_error(error):
    print(f"🚨 Time Capsule Check Loop Error: {error}")

# --------------------------------------------------------
# 📼 BACKGROUND TASK: MEMORY BANK RESURFACING
# --------------------------------------------------------
@tasks.loop(hours=3.0)
async def memory_resurface_loop():
    for guild in bot.guilds:
        gid = str(guild.id)
        saved = memory_bank.get(gid, [])
        if not saved:
            continue
        if random.random() > 0.35:
            continue

        memory = random.choice(saved)
        channel = guild.get_channel(memory["channel_id"])
        if not channel:
            continue

        embed = discord.Embed(
            title="📼 A memory resurfaces...",
            description=memory["content"],
            color=discord.Color.dark_purple(),
            timestamp=datetime.fromisoformat(memory["saved_at"])
        )
        embed.set_footer(text=f"Originally said by {memory['author_name']}")
        try:
            await channel.send(embed=embed, view=None)
        except Exception as e:
            print(f"⚠️ Failed to resurface memory in {guild.name}: {e}")

@memory_resurface_loop.error
async def memory_resurface_loop_error(error):
    print(f"🚨 Memory Resurface Loop Error: {error}")

# --------------------------------------------------------
# 💀 BACKGROUND TASK: RANDOM BRUTAL ROAST
# --------------------------------------------------------
async def find_most_active_channel(guild: discord.Guild) -> Optional[discord.TextChannel]:
    """Picks whichever text channel had the most recent message, using each
    channel's last_message_id snowflake timestamp -- no extra API calls
    needed. Works on any server automatically, no setup required (unlike
    relying on a single configured/hardcoded channel that only exists on
    one particular server)."""
    best_channel = None
    best_time = None
    for channel in guild.text_channels:
        perms = channel.permissions_for(guild.me)
        if not (perms.view_channel and perms.read_message_history and perms.send_messages):
            continue
        if channel.last_message_id is None:
            continue
        try:
            msg_time = discord.utils.snowflake_time(channel.last_message_id)
        except Exception:
            continue
        if best_time is None or msg_time > best_time:
            best_time = msg_time
            best_channel = channel
    return best_channel

@tasks.loop(hours=6.0)
async def random_roast_loop():
    """Runs every 6 hours, independently for every server the bot is in --
    picks a random recent chatter in that server's most active channel and
    fires off a genuinely savage AI roast based on their ACTUAL recent
    messages. UPDATED: mods can now turn this off per-server with
    ?disableroast (see the command below) -- servers that have done so are
    skipped entirely here."""
    for guild in bot.guilds:
        if server_config.get(str(guild.id), {}).get("random_roast_disabled"):
            continue

        channel = await find_most_active_channel(guild)
        if not channel:
            continue

        try:
            history = [m async for m in channel.history(limit=100) if not m.author.bot and m.content.strip()]
        except Exception:
            continue

        by_author = defaultdict(list)
        for m in history:
            by_author[m.author.id].append(m.content)

        candidates = {uid: msgs for uid, msgs in by_author.items() if len(msgs) >= 3}
        if not candidates:
            continue

        target_id = random.choice(list(candidates.keys()))
        target_member = guild.get_member(target_id)
        if not target_member or target_member.bot:
            continue

        samples = " | ".join(candidates[target_id][-10:])[:700]
        prompt = (
            f"Here are real recent messages from a Discord user named {target_member.display_name}:\n{samples}\n\n"
            "Write ONE brutal, savage, no-holds-barred roast (2-3 sentences), grounded specifically in the "
            "actual content/vibe/energy of these messages -- reference something concrete from what they "
            "said. Go hard, be genuinely witty and cutting, like a celebrity comedy roast. Do NOT attack "
            "their race, ethnicity, religion, sexual orientation, gender identity, disability, or any other "
            "protected characteristic, and don't use slurs or genuinely hateful language -- roast their "
            "chat behavior/personality/vibe, not who they are as a person. Output only the roast."
        )
        try:
            roast_text = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"⚠️ Random roast generation failed: {e}")
            continue

        embed = discord.Embed(
            title=f"💀 Random Roast: {target_member.display_name}",
            description=truncate_text(roast_text.strip(), 1000),
            color=discord.Color.dark_red()
        )
        embed.set_thumbnail(url=target_member.display_avatar.url)
        try:
            await channel.send(embed=embed)
        except Exception as e:
            print(f"⚠️ Failed to post random roast: {e}")

@random_roast_loop.error
async def random_roast_loop_error(error):
    print(f"🚨 Random Roast Loop Error: {error}")

# --------------------------------------------------------
# 🏆 BACKGROUND TASK: WEEKLY LEADERBOARD
# --------------------------------------------------------
@tasks.loop(time=time(hour=0, minute=0, tzinfo=IST_TIMEZONE))
async def weekly_leaderboard_announcement():
    now_ist = datetime.now(IST_TIMEZONE)
    if now_ist.weekday() != 6: return 

    global message_counts
    for guild_id_str, counts in message_counts.items():
        guild = bot.get_guild(int(guild_id_str))
        if not guild: continue

        channel = get_configured_channel(guild, "leaderboard_channel_id", LEADERBOARD_CHANNEL_ID, LEADERBOARD_CHANNEL_NAME)
        if not channel: continue 
            
        user_counts = {k: v for k, v in counts.items() if guild.get_member(int(k)) and not guild.get_member(int(k)).bot}
        
        if not user_counts:
            await channel.send("The weekly message leaderboard reset has occurred, but no active user messages were found this week.")
            continue

        # NEW: Personalized Weekly Digest DM -- before anything else, give each
        # active (non-opted-out) member their own AI-written recap of their
        # week. Fire-and-forget tasks so one slow/failed DM never blocks the
        # rest of the leaderboard processing for this guild.
        sorted_for_rank = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)
        for uid_str, count in user_counts.items():
            uid = int(uid_str)
            if uid in digest_optout:
                continue
            digest_member = guild.get_member(uid)
            if not digest_member:
                continue
            digest_rank = next((i + 1 for i, (u, c) in enumerate(sorted_for_rank) if u == uid_str), None)
            bot.loop.create_task(send_weekly_digest_dm(digest_member, guild, count, digest_rank, len(user_counts)))

        winner_id_str, max_messages = max(user_counts.items(), key=lambda item: item[1])
        winner = guild.get_member(int(winner_id_str))

        if not winner or max_messages == 0:
            await channel.send("The weekly message leaderboard reset has occurred, but no active winner was found this week.")
            continue

        winner_role = discord.utils.get(guild.roles, name=WINNER_ROLE_NAME)
        if not winner_role:
            try:
                winner_role = await guild.create_role(name=WINNER_ROLE_NAME, color=discord.Color.yellow(), reason="Weekly Activity Leaderboard Role")
                await channel.send(f"**ADMIN NOTE:** The required role **{WINNER_ROLE_NAME}** was created automatically. Please adjust its permissions/position.")
            except discord.Forbidden:
                await channel.send(f"❌ Cannot assign **{WINNER_ROLE_NAME}**. Bot lacks `manage_roles` permission or the role is too high.")
                continue

        for member in guild.members:
            if winner_role in member.roles and member.id != winner.id:
                try:
                    await member.remove_roles(winner_role, reason="Previous Chatterbox winner cleanup.")
                except discord.Forbidden:
                    pass

        try:
            await winner.add_roles(winner_role, reason="Weekly Activity Champion: Highest message count.")
            removal_delay = 60 * 60 * 24 * 7 
            bot.loop.create_task(
                remove_winner_role_after_delay(guild.id, winner.id, winner_role.id, removal_delay, channel.id)
            )

        except discord.Forbidden:
            await channel.send(f"❌ Failed to assign **{WINNER_ROLE_NAME}** to {winner.mention}. Check bot role hierarchy.")
            continue
        
        embed = discord.Embed(
            title=f"🏆 WEEKLY ACTIVITY CHAMPION! 🏆",
            description=f"Our chat king/queen for the week is...",
            color=discord.Color.gold()
        )
        embed.add_field(name=f"🥇 The Winner: {winner.display_name} 🥇", value=f"They sent a massive **{max_messages}** messages this week!", inline=False)
        embed.add_field(name=f"👑 Reward:", value=f"They have won the temporary custom role: **{WINNER_ROLE_NAME}**!", inline=False)
        embed.set_thumbnail(url=winner.display_avatar.url)
        embed.set_footer(text=f"Role will expire in 7 days. Counts reset for the next week! Start chatting!")
        await channel.send(f"🎉 **@everyone** 🎉", embed=embed)
        
        message_counts[guild_id_str] = {} 
        
    save_data(message_counts, MESSAGES_FILE)

@weekly_leaderboard_announcement.error
async def weekly_leaderboard_announcement_error(error):
    print(f"🚨 Weekly Leaderboard Task Error: {error}")

async def send_weekly_digest_dm(member: discord.Member, guild: discord.Guild, msg_count: int, rank: Optional[int], total_active: int):
    """Builds and DMs one member their personalized weekly recap. Silently
    gives up if their DMs are closed -- nothing else needs to know."""
    context = _birthday_context(member, guild)
    prompt = (
        f"Write a short, warm, personal 'weekly recap' DM (4-6 sentences) for a Discord user, based on "
        f"this context about their week in the server:\n{context}\n"
        f"They ranked #{rank} out of {total_active} active chatters this week. "
        "Make it feel personal and specific to them, not generic. Sprinkle in a couple relevant emojis. "
        "Sign off warmly. Output only the message, nothing else."
    )
    try:
        text = await asyncio.to_thread(get_groq_text, prompt)
    except Exception as e:
        print(f"⚠️ Weekly digest generation failed for {member}: {e}")
        return

    embed = discord.Embed(
        title=f"🗞️ Your Week in {guild.name}",
        description=text,
        color=discord.Color.blurple(),
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_footer(text="Opt out any time with ?digestoptout")
    try:
        await member.send(embed=embed)
    except Exception:
        pass  # DMs closed -- nothing more to do

# --------------------------------------------------------
# 🎂 BACKGROUND TASK: DAILY BIRTHDAY CHECK
# --------------------------------------------------------
@tasks.loop(time=time(hour=0, minute=0, tzinfo=IST_TIMEZONE))
async def birthday_check_loop():
    now_ist = datetime.now(IST_TIMEZONE)
    today_day, today_month = now_ist.day, now_ist.month

    for user_id_str, bday in list(birthdays.items()):
        if bday.get("day") != today_day or bday.get("month") != today_month:
            continue

        user_id = int(user_id_str)
        for guild in bot.guilds:
            member = guild.get_member(user_id)
            if member:
                issues = await celebrate_birthday(guild, member, bday)
                if issues:
                    print(f"🚨 Birthday celebration issues for {member} in {guild.name}: {issues}")
                    mod_log_channel = get_mod_log_channel(guild)
                    if mod_log_channel:
                        try:
                            await mod_log_channel.send(
                                f"🎂 Birthday celebration for {member.mention} had some issues:\n" + "\n".join(issues)
                            )
                        except Exception:
                            pass

@birthday_check_loop.error
async def birthday_check_loop_error(error):
    print(f"🚨 Birthday Check Loop Error: {error}")

# --------------------------------------------------------
# 🎂 BACKGROUND TASK: BIRTHDAY ROLE CLEANUP BACKSTOP
# --------------------------------------------------------
@tasks.loop(minutes=15.0)
async def birthday_role_cleanup_loop():
    if not birthday_role_cleanups:
        return

    now = datetime.now(timezone.utc)
    for role_id_str, record in list(birthday_role_cleanups.items()):
        try:
            expires_at = datetime.fromisoformat(record["expires_at"])
        except Exception:
            birthday_role_cleanups.pop(role_id_str, None)
            save_data(birthday_role_cleanups, BIRTHDAY_ROLE_CLEANUP_FILE)
            continue

        if now >= expires_at:
            await perform_birthday_role_cleanup(record["guild_id"], record["member_id"], int(role_id_str))

@birthday_role_cleanup_loop.error
async def birthday_role_cleanup_loop_error(error):
    print(f"🚨 Birthday Role Cleanup Loop Error: {error}")

@tasks.loop(minutes=15.0)
async def temp_role_cleanup_loop():
    if not temp_roles:
        return

    now = datetime.now(timezone.utc)
    for role_id_str, record in list(temp_roles.items()):
        try:
            expires_at = datetime.fromisoformat(record["expires_at"])
        except Exception:
            temp_roles.pop(role_id_str, None)
            save_data(temp_roles, TEMP_ROLES_FILE)
            continue

        if now >= expires_at:
            await perform_temp_role_cleanup(record["guild_id"], record["member_id"], int(role_id_str), record.get("delete_role", True))

@temp_role_cleanup_loop.error
async def temp_role_cleanup_loop_error(error):
    print(f"🚨 Temp Role Cleanup Loop Error: {error}")

# --------------------------------------------------------
# 🤖 BOT EVENTS
# --------------------------------------------------------

# --------------------------------------------------------
# 🎭 DYNAMIC ROLE MENU (?setup_roles, ?addnewcategory, ?addnewsetup)
# --------------------------------------------------------
DEFAULT_ROLE_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age Group": ["18-", "18+"],
    "Pronouns": ["He/Him", "She/Her", "They/Them", "Ask Me"],
    "Games": ["Roblox", "Minecraft", "Valorant", "BGMI", "CS2", "GTA", "Fortnite", "Call Of Duty", "Mobile Legends"],
    "Notification Pings": ["Announcements Ping", "Giveaway Ping", "Events Ping"],
}
CATEGORY_EMOJIS = {"Gender": "🚻", "Age Group": "🎂", "Pronouns": "🗣️", "Games": "🎮", "Notification Pings": "🔔"}
DEFAULT_CATEGORY_EMOJI = "✨"
# NEW: cycled per-option so any custom role a mod adds via ?addnewsetup still
# gets an emoji in the dropdown, matching the built-in categories' style.
OPTION_EMOJI_CYCLE = ["🔹", "🔸", "⭐", "🌟", "💠", "🔺", "🔻", "🔷", "🔶", "♦️", "🔘", "⚪", "⚫", "🟣", "🟢", "🔵", "🟡", "🟠", "🔴", "🟤"]

def get_guild_role_categories(guild_id: int) -> dict:
    gid = str(guild_id)
    if gid not in role_menu_config:
        # Seeds with the original built-in categories so existing servers
        # keep working exactly the same the first time this runs for them.
        role_menu_config[gid] = {name: list(roles) for name, roles in DEFAULT_ROLE_CATEGORIES.items()}
        save_data(role_menu_config, ROLE_MENU_CONFIG_FILE)
    return role_menu_config[gid]

def category_emoji(category: str) -> str:
    return CATEGORY_EMOJIS.get(category, DEFAULT_CATEGORY_EMOJI)

def option_emoji(index: int) -> str:
    return OPTION_EMOJI_CYCLE[index % len(OPTION_EMOJI_CYCLE)]

def slugify_category(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")[:60] or "cat"

def chunk_categories(categories: dict, size: int = 5) -> list[dict]:
    """Discord caps a single message's view at 5 action rows -- each dropdown
    takes one row, so more than 5 categories need multiple messages/views."""
    items = list(categories.items())
    return [dict(items[i:i + size]) for i in range(0, len(items), size)]

class DynamicRolePicker(ui.View):
    """One Select dropdown per role category, built from whatever's currently
    configured for the guild -- mods can add categories/roles at runtime via
    ?addnewcategory / ?addnewsetup instead of these being hardcoded in code."""
    def __init__(self, categories: dict[str, list[str]]):
        super().__init__(timeout=None)
        for cat_name, role_names in categories.items():
            if not role_names:
                continue
            slug = slugify_category(cat_name)
            options = [
                discord.SelectOption(label=name[:100], emoji=option_emoji(i))
                for i, name in enumerate(role_names[:25])  # Discord caps a Select at 25 options
            ]
            select = ui.Select(
                custom_id=f"roles_dyn_{slug}",
                placeholder=f"{category_emoji(cat_name)} {cat_name} (pick any)",
                min_values=0, max_values=len(options),
                options=options,
            )

            async def _callback(interaction: Interaction, select=select, role_names=role_names):
                await self._apply_selection(interaction, role_names, select.values)

            select.callback = _callback
            self.add_item(select)

    async def _get_or_create_role(self, guild: discord.Guild, name: str) -> Optional[discord.Role]:
        role = discord.utils.get(guild.roles, name=name)
        if role:
            return role
        try:
            # NEW: colourless by request -- keeps auto-created self-roles neutral.
            return await guild.create_role(name=name, color=discord.Color.default(), reason="Auto-created for self-role menu")
        except discord.Forbidden:
            return None

    async def _apply_selection(self, interaction: Interaction, all_role_names: list[str], selected_names: list[str]):
        member = interaction.user
        guild = interaction.guild
        to_add, to_remove = [], []

        for name in all_role_names:
            role = await self._get_or_create_role(guild, name)
            if not role:
                continue
            has_it = role in member.roles
            wants_it = name in selected_names
            if wants_it and not has_it:
                to_add.append(role)
            elif has_it and not wants_it:
                to_remove.append(role)

        try:
            if to_add:
                await member.add_roles(*to_add, reason="Self-role menu")
            if to_remove:
                await member.remove_roles(*to_remove, reason="Self-role menu")
        except discord.Forbidden:
            return await interaction.response.send_message(
                "❌ I don't have permission to manage one of those roles -- my role needs to be positioned above them in the role list.",
                ephemeral=True
            )

        summary = f"✅ Updated! You now have: **{', '.join(selected_names)}**" if selected_names else "✅ Cleared -- you have none of these roles now."
        await interaction.response.send_message(summary, ephemeral=True)

class CreateMenu(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(
        label="Poem",
        emoji="✍️",
        style=discord.ButtonStyle.primary,
        row=0,
        custom_id="create_poem"
    )
    async def poem(self, interaction: discord.Interaction, button: discord.ui.Button):
        thread = await interaction.channel.create_thread(
            name=f"Poem-{interaction.user.name}",
            auto_archive_duration=60,
            type=discord.ChannelType.public_thread
        )

        await interaction.response.send_message(
            f"✅ Poem thread created: {thread.mention}", ephemeral=True
        )

        words = await asyncio.to_thread(
            get_groq_text, "Give 5 random poetic words."
        )

        await thread.send(
            f"Welcome {interaction.user.mention}! Write a poem using:\n**{words}**"
        )

    @discord.ui.button(
        label="Riddle",
        emoji="🧩",
        style=discord.ButtonStyle.success,
        row=0,
        custom_id="create_riddle"
    )
    async def riddle(self, interaction: discord.Interaction, button: discord.ui.Button):
        thread = await interaction.channel.create_thread(
            name=f"Case-{interaction.user.name}",
            auto_archive_duration=60,
            type=discord.ChannelType.public_thread
        )

        await interaction.response.send_message(
            f"🕵️ Case started in: {thread.mention}", ephemeral=True
        )

        try:
            case = await asyncio.to_thread(
                get_groq_text,
                "Generate a unique, short detective riddle."
            )

            if not case or len(case.strip()) == 0:
                case = "I speak without a mouth and hear without ears. What am I?"

        except Exception as e:
            print(f"Riddle Error: {e}")
            case = "I speak without a mouth and hear without ears. What am I?"

        await safe_send(thread, f"🔍 **THE MYSTERY:**\n{case}")

        bot.loop.create_task(self.riddle_marathon(thread, interaction.user))

    @discord.ui.button(
        label="Song",
        emoji="🎤",
        style=discord.ButtonStyle.danger,
        row=0,
        custom_id="create_song"
    )
    async def song(self, interaction: discord.Interaction, button: discord.ui.Button):
        thread = await interaction.channel.create_thread(
            name=f"Song-{interaction.user.name}",
            auto_archive_duration=60,
            type=discord.ChannelType.public_thread
        )

        await interaction.response.send_message(
            f"🎤 Studio opened: {thread.mention}", ephemeral=True
        )

        await thread.send(
            "Upload your song (mp3/wav) here! I'll give you a professional critique."
        )

        bot.loop.create_task(delete_thread_later(thread, 86400))

    async def riddle_marathon(self, thread, user):
        await thread.send(
            "*7-Day Challenge Started!*\nSolve the riddle above.\nHints coming every 24 hours."
        )

        for i in range(1, 7):
            await asyncio.sleep(86400)
            if thread:
                await thread.send(f"*Day {i+1} Hint:*")

        await asyncio.sleep(86400)
        await thread.send("Time's up! Thread deleting...")
        await asyncio.sleep(10)
        await thread.delete()
        
@bot.event
async def on_ready():
    print(f'Bot is ready! Logged in as {bot.user}')
    global current_bot_status, current_bot_activity
    current_bot_activity = discord.Game(name="?help | AI Chat")
    await bot.change_presence(status=current_bot_status, activity=current_bot_activity)

    try:
        bot.add_view(CreateMenu())
        # NEW: role menu is now per-guild and dynamic, so each guild's
        # current categories (chunked to Discord's 5-rows-per-view cap) get
        # their own persistent view registered, instead of one fixed RolePicker.
        for g in bot.guilds:
            categories = get_guild_role_categories(g.id)
            non_empty = {name: roles for name, roles in categories.items() if roles}
            for chunk in chunk_categories(non_empty, size=5):
                bot.add_view(DynamicRolePicker(chunk))
        # NEW: ?bnstory's genre-select and continue-or-stop views are
        # persistent (custom_id-based, no per-instance state) so they keep
        # working across a restart/redeploy -- see the ?bnstory section.
        bot.add_view(StoryGenreView())
        bot.add_view(StoryContinueView())
    except Exception as e:
        print(f"🚨 Failed to register persistent views: {e}")

    if bot.http_session is None:
        bot.http_session = aiohttp.ClientSession()

    if not reminder_check_loop.is_running():
        reminder_check_loop.start()

    if not timecapsule_check_loop.is_running():
        timecapsule_check_loop.start()

    if not memory_resurface_loop.is_running():
        memory_resurface_loop.start()

    if not random_roast_loop.is_running():
        random_roast_loop.start()

    if not weekly_leaderboard_announcement.is_running():
        weekly_leaderboard_announcement.start()

    if not birthday_check_loop.is_running():
        birthday_check_loop.start()

    if not birthday_role_cleanup_loop.is_running():
        birthday_role_cleanup_loop.start()

    if not temp_role_cleanup_loop.is_running():
        temp_role_cleanup_loop.start()

    # NEW: catches ?bnstory sessions whose timer ran out while idle (see the
    # ?bnstory section for why this is needed alongside the inline check).
    if not story_expiry_check_loop.is_running():
        story_expiry_check_loop.start()

    # NEW: runs delayed/auto-revert actions from the natural-language control
    # channel (?setcontrolchannel) -- e.g. "unjail @user after 10 mins".
    if not scheduled_control_action_loop.is_running():
        scheduled_control_action_loop.start()

    # NEW: registers /av, /rate, /afk (and any other slash commands) with
    # Discord. Global sync can take up to ~1 hour to show up everywhere the
    # first time -- that's a Discord-side propagation delay, not a bug.
    global slash_commands_synced
    if not slash_commands_synced:
        try:
            await bot.tree.sync()
            slash_commands_synced = True
            print("✅ Slash commands synced.")
        except Exception as e:
            print(f"🚨 Slash command sync failed: {e}")

    # NEW: seed active_voice_sessions with anyone already sitting in a voice
    # channel when the bot (re)starts, so their time counts from now instead
    # of only starting once they next join/leave/move channels.
    startup_time = datetime.now(timezone.utc)
    for g in bot.guilds:
        for vc in g.voice_channels:
            for vc_member in vc.members:
                if vc_member.bot:
                    continue
                active_voice_sessions.setdefault((g.id, vc.id), {})[vc_member.id] = startup_time

def record_voice_overlap(guild_id: int, channel_id: int, leaving_user_id: int, now: datetime):
    """When someone leaves/moves out of a voice channel, credits the time they
    overlapped with every other member still in that session toward the
    pairwise VC-time-together total (used by ?twin)."""
    session = active_voice_sessions.get((guild_id, channel_id), {})
    my_join = session.get(leaving_user_id)
    if my_join is None:
        return

    gid_str = str(guild_id)
    changed = False
    for other_id, other_join in session.items():
        if other_id == leaving_user_id:
            continue
        overlap_start = max(my_join, other_join)
        overlap_seconds = (now - overlap_start).total_seconds()
        if overlap_seconds > 0:
            pair_key = "-".join(sorted([str(leaving_user_id), str(other_id)]))
            voice_time_together.setdefault(gid_str, {})
            voice_time_together[gid_str][pair_key] = voice_time_together[gid_str].get(pair_key, 0) + overlap_seconds
            changed = True

    if changed:
        save_data(voice_time_together, VOICE_TIME_FILE)

@bot.event
async def on_voice_state_update(member, before, after):
    if member.bot:
        return
    now = datetime.now(timezone.utc)

    if before.channel != after.channel:
        if before.channel is not None:
            key = (member.guild.id, before.channel.id)
            if key in active_voice_sessions and member.id in active_voice_sessions[key]:
                record_voice_overlap(member.guild.id, before.channel.id, member.id, now)
                del active_voice_sessions[key][member.id]
                if not active_voice_sessions[key]:
                    del active_voice_sessions[key]

        if after.channel is not None:
            key = (member.guild.id, after.channel.id)
            active_voice_sessions.setdefault(key, {})[member.id] = now

@bot.event
async def on_member_join(member):
    gid = str(member.guild.id)
    uid = str(member.id)
    now_iso = datetime.now(timezone.utc).isoformat()

    if gid not in member_history:
        member_history[gid] = {}
    record = member_history[gid].get(uid, {})

    is_rejoin = "current_joined" in record

    if "first_joined" not in record:
        record["first_joined"] = now_iso
    record["current_joined"] = now_iso
    record["join_count"] = record.get("join_count", 0) + 1

    member_history[gid][uid] = record
    save_data(member_history, MEMBER_HISTORY_FILE)

    # NEW: if this member was kicked and that kick was later undone via
    # ?undo, restore the roles they had before the kick now that they're
    # back (pending_kick_restores is in-memory only -- see undo_kick).
    restore_key = (member.guild.id, member.id)
    if restore_key in pending_kick_restores:
        role_ids = pending_kick_restores.pop(restore_key)
        roles_to_add = [r for r in (member.guild.get_role(rid) for rid in role_ids) if r]
        if roles_to_add:
            try:
                await member.add_roles(*roles_to_add, reason="Restoring roles after an undone kick")
            except Exception as e:
                print(f"⚠️ Failed to restore roles after undone kick for {member}: {e}")

    # UPDATED: was a single hardcoded channel ID, so welcome messages only ever
    # posted on your main server no matter which server someone joined. Now each
    # server configures its own with `?setchannel welcome #channel`. If a server
    # hasn't set one, the welcome message is just skipped (no error, no wrong-server spam).
    channel = get_configured_channel(member.guild, "welcome_channel_id")
    
    if channel:
        embed = discord.Embed(
            title="Welcome to the Server! 🎉",
            description=f"Welcome {member.mention}! We're glad to have you here.",
            color=discord.Color.green(),
            timestamp=datetime.now(timezone.utc)
        )
        embed.set_thumbnail(url=member.display_avatar.url)
        embed.add_field(name="Member Count", value=f"You are our {member.guild.member_count}th member!", inline=False)

        if is_rejoin and record.get("last_left"):
            last_left_dt = datetime.fromisoformat(record["last_left"])
            embed.add_field(
                name="Welcome Back!",
                value=f"You previously left this server on {last_left_dt.strftime('%Y-%m-%d %H:%M UTC')}.",
                inline=False
            )

        embed.set_footer(text=f"ID: {member.id}")
        
        await channel.send(content=f"Hey {member.mention}, welcome!", embed=embed)

@bot.event
async def on_member_remove(member):
    gid = str(member.guild.id)
    uid = str(member.id)

    if gid not in member_history:
        member_history[gid] = {}
    record = member_history[gid].get(uid, {})
    record["last_left"] = datetime.now(timezone.utc).isoformat()
    member_history[gid][uid] = record
    save_data(member_history, MEMBER_HISTORY_FILE)

    # NEW: Auto-Eulogy -- if this server has a channel configured for it,
    # post an AI-written send-off based on the member's real stats.
    if not member.bot:
        eulogy_channel = get_configured_channel(member.guild, "eulogy_channel_id")
        if eulogy_channel:
            bot.loop.create_task(post_auto_eulogy(member, member.guild, eulogy_channel))

async def post_auto_eulogy(member: discord.Member, guild: discord.Guild, channel):
    """Builds and posts a lighthearted 'in memoriam' for a member who just
    left, using their real tracked stats. Runs as a background task so it
    never delays/blocks the on_member_remove event itself."""
    gid, uid = str(guild.id), str(member.id)
    msg_count = message_counts.get(gid, {}).get(uid, 0)
    reps = reputation.get(uid, 0)
    badges = get_earned_badges(member, guild)
    record = member_history.get(gid, {}).get(uid, {})
    join_count = record.get("join_count", 1)
    was_married = uid in get_guild_marriages(guild.id)
    times_married = get_guild_marriage_stats(guild.id).get(uid, 0)

    tenure_note = "unknown"
    first_joined = record.get("first_joined")
    if first_joined:
        try:
            days = (datetime.now(timezone.utc) - datetime.fromisoformat(first_joined)).days
            tenure_note = f"{days} days"
        except Exception:
            pass

    context = (
        f"Display name: {member.display_name}\n"
        f"Time in server: {tenure_note}\n"
        f"Times joined (including rejoins): {join_count}\n"
        f"Messages sent (most recent week's count): {msg_count}\n"
        f"Reputation points: {reps}\n"
        f"Badges earned: {', '.join(b[1] for b in badges) if badges else 'none'}\n"
        f"Married when they left: {'yes' if was_married else 'no'}\n"
        f"Total times married (all-time): {times_married}\n"
    )
    prompt = (
        "Write a short, warm, slightly funny 'in memoriam' post (4-6 sentences) for a Discord member "
        f"who just left the server, based on this real data about them:\n{context}\n"
        "Treat it like a lighthearted send-off, not morbid -- celebrate their time here. Mention their "
        "name naturally. Output only the eulogy, nothing else."
    )
    try:
        eulogy_text = await asyncio.to_thread(get_groq_text, prompt)
    except Exception as e:
        print(f"⚠️ Eulogy generation failed, using fallback: {e}")
        eulogy_text = f"{member.display_name} has left the server. Thanks for the time you spent here. 🕊️"

    embed = discord.Embed(
        title="🕊️ In Memoriam",
        description=eulogy_text,
        color=discord.Color.dark_grey(),
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    embed.set_footer(text="Auto-generated based on their time here")
    try:
        await channel.send(embed=embed)
    except Exception as e:
        print(f"⚠️ Failed to post auto-eulogy: {e}")

# NEW: keeps private_channels.json clean if a private channel gets deleted
# some other way (e.g. an Administrator deletes it directly via Discord's UI,
# bypassing ?deletepvtchannel) -- otherwise the stale record would just sit
# there forever pointing at a channel that no longer exists.
@bot.event
async def on_guild_channel_delete(channel):
    if str(channel.id) in private_channels:
        private_channels.pop(str(channel.id), None)
        save_data(private_channels, PRIVATE_CHANNELS_FILE)

# NEW: keeps story_sessions.json clean if a ?bnstory thread gets deleted some
# other way (archived-and-auto-purged, a mod deletes it manually, etc.) --
# otherwise a stale entry could later collide with a reused/duplicate thread ID.
@bot.event
async def on_thread_delete(thread):
    tid = str(thread.id)
    if tid in story_sessions:
        story_sessions.pop(tid, None)
        save_story_sessions()

@bot.event
async def on_message_delete(message):
    
    if message.author.bot or message.guild is None:
        return

    now = datetime.now(timezone.utc)

    # NEW: logs messages a MODERATOR deleted (via Discord's audit log) to the
    # mod-log channel. Deliberately only fires when the audit log confirms a
    # mod did it -- Discord doesn't create an audit entry when someone deletes
    # their own message, so this naturally excludes normal self-deletions
    # instead of flooding mod-log with every message anyone removes themselves.
    if message.content.strip():
        mod_log_channel = get_mod_log_channel(message.guild)
        if mod_log_channel:
            deleted_by = None
            try:
                async for entry in message.guild.audit_logs(limit=5, action=discord.AuditLogAction.message_delete):
                    if (entry.target and entry.target.id == message.author.id
                            and entry.extra.channel.id == message.channel.id
                            and (now - entry.created_at.replace(tzinfo=timezone.utc)).total_seconds() < 10):
                        deleted_by = entry.user
                        break
            except discord.Forbidden:
                pass  # bot lacks View Audit Log -- silently skip, can't attribute
            except Exception as e:
                print(f"⚠️ Audit log lookup for message delete failed: {e}")

            if deleted_by:
                embed = discord.Embed(
                    title="🗑️ Message Deleted by Moderator",
                    color=discord.Color.dark_grey(),
                    timestamp=now
                )
                embed.add_field(name="Author", value=f"{message.author.mention} ({message.author.id})", inline=True)
                embed.add_field(name="Channel", value=message.channel.mention, inline=True)
                embed.add_field(name="Deleted By", value=deleted_by.mention, inline=True)
                embed.add_field(name="Content", value=message.content[:1000], inline=False)
                try:
                    await mod_log_channel.send(embed=embed)
                except Exception as e:
                    print(f"⚠️ Failed to post message-delete log: {e}")

    if not message.mentions:
        return

    if (now - message.created_at).total_seconds() > 120:
        return

    # UPDATED: was a single hardcoded channel ID (only worked on one server).
    # Now configurable per-server with `?setchannel ghostlog #channel`.
    log_channel = get_configured_channel(message.guild, "ghost_log_channel_id")
    if log_channel:
        embed = discord.Embed(
            title="👻 Ghost Ping Detected!",
            color=discord.Color.red(),
            timestamp=now
        )
        embed.add_field(name="Author", value=f"{message.author.mention} ({message.author.id})", inline=True)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        
        pinged_users = ", ".join([user.mention for user in message.mentions])
        embed.add_field(name="Users Targeted", value=pinged_users, inline=False)
        
        if message.content:
            embed.add_field(name="Deleted Content", value=message.content, inline=False)

        await log_channel.send(embed=embed)

@bot.event
async def on_message_edit(before, after):
    if before.mentions and not after.mentions:
        if before.author.bot or before.guild is None: return

        # UPDATED: per-server configurable, see on_message_delete above.
        log_channel = get_configured_channel(before.guild, "ghost_log_channel_id")
        if log_channel:
            embed = discord.Embed(
                title="📝 Ghost Ping (Edited)",
                color=discord.Color.orange(),
                timestamp=datetime.now(timezone.utc)
            )
            embed.add_field(name="Author", value=before.author.mention, inline=True)
            embed.add_field(name="Channel", value=before.channel.mention, inline=True)
            
            pinged_users = ", ".join([user.mention for user in before.mentions])
            embed.add_field(name="Pings Removed", value=pinged_users, inline=False)
            embed.add_field(name="Original Content", value=before.content, inline=False)

            await log_channel.send(embed=embed)

# --------------------------------------------------------
# 🌟 HALL OF FAME (STARBOARD)
# --------------------------------------------------------
@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.guild_id is None:
        return

    guild = bot.get_guild(payload.guild_id)
    if not guild:
        return

    if str(payload.emoji) == "📌":
        member = payload.member
        if member and not member.bot and member.guild_permissions.manage_messages:
            try:
                channel = guild.get_channel(payload.channel_id) or await bot.fetch_channel(payload.channel_id)
                message = await channel.fetch_message(payload.message_id)
                if not message.pinned:
                    await message.pin(reason=f"Pinned by {member} via 📌 reaction")
            except discord.Forbidden:
                print(f"⚠️ Missing permissions to pin messages in {guild.name}.")
            except discord.HTTPException as e:
                print(f"⚠️ Couldn't pin message (channel may already have 50 pins): {e}")
            except Exception as e:
                print(f"⚠️ Reaction-to-pin failed: {e}")

    if str(payload.emoji) == "📝":
        try:
            channel = guild.get_channel(payload.channel_id) or await bot.fetch_channel(payload.channel_id)
            message = await channel.fetch_message(payload.message_id)
            if message.content and not message.author.bot:
                gid = str(guild.id)
                memory_bank.setdefault(gid, [])
                if not any(m["message_id"] == message.id for m in memory_bank[gid]):
                    memory_bank[gid].append({
                        "content": message.content,
                        "author_id": message.author.id,
                        "author_name": message.author.display_name,
                        "channel_id": message.channel.id,
                        "message_id": message.id,
                        "jump_url": message.jump_url,
                        "saved_at": datetime.now(timezone.utc).isoformat()
                    })
                    save_data(memory_bank, MEMORY_BANK_FILE)
        except Exception as e:
            print(f"⚠️ Failed to save memory: {e}")

@bot.event
async def on_message(message):
    if message.author == bot.user or message.guild is None:
        await bot.process_commands(message)
        return

    content = message.content
    content_lower = content.lower()

    # BUG FIX: was keyed by user_id alone, so going AFK on one server and
    # then posting on a DIFFERENT server (where the bot is also in) cleared
    # your AFK status everywhere, not just the server you actually spoke in.
    # Now keyed per (guild, user) so each server's AFK status is independent.
    afk_key = f"{message.guild.id}_{message.author.id}"
    if afk_key in afk_users:
        del afk_users[afk_key]
        save_data(afk_users, AFK_FILE)
        await message.channel.send(f"👋 Welcome back, {message.author.mention}! You're no longer AFK.", delete_after=5)

    # NEW: natural-language control channel -- routes plain-English mod
    # instructions to the AI parser + dispatcher instead of falling through
    # to the rest of the normal pipeline (spam tracking, triggers, etc).
    control_channel_id = server_config.get(str(message.guild.id), {}).get("control_channel_id")
    if control_channel_id and str(message.channel.id) == control_channel_id and not content.startswith(bot.command_prefix):
        if message.author.guild_permissions.administrator:
            await handle_control_instruction(message)
        else:
            try:
                await message.delete()
            except Exception:
                pass
            await message.channel.send(f"❌ {message.author.mention}, only server Administrators can issue instructions in this channel.", delete_after=10)
        return

    if content.strip() == bot.command_prefix:
        suggestions = ["talk", "rate", "help", "ping", "afk", "leaderboard", "profile", "create"]
        suggestion_text = "  •  ".join(f"`{bot.command_prefix}{cmd}`" for cmd in suggestions)
        await message.channel.send(
            f"👋 Try one of these, or use `{bot.command_prefix}help` to see everything:\n{suggestion_text}",
            delete_after=15
        )
        return

    # NEW: ?bnstory active roleplay -- if this message is inside an active
    # story thread, route it to the AI instead of falling through to the
    # rest of the normal message-handling pipeline below (spam tracking,
    # highlights, restricted words, etc. don't apply inside a story).
    if isinstance(message.channel, discord.Thread):
        story_session = story_sessions.get(str(message.channel.id))
        if story_session and story_session.get("state") == "active" and not content.startswith(bot.command_prefix):
            await handle_story_message(message, story_session)
            await bot.process_commands(message)
            return

    if (
        message.reference is not None
        and message.reference.message_id in active_talk_messages
        and active_talk_messages[message.reference.message_id] == message.author.id
        and not content.startswith(bot.command_prefix)
        and content.strip()
        and (message.guild is None or is_talk_allowed_in_channel(message.guild, message.channel.id))
    ):
        async with message.channel.typing():
            response = await asyncio.to_thread(get_groq_chat_response, message.author.id, content)
            sent_messages = await safe_send(message.channel, response, reply_to=message)
            for m in sent_messages:
                track_talk_message(m.id, message.author.id)
        return

    if not message.content.startswith(bot.command_prefix):
        if restricted_words:
            for word in restricted_words:
                if word in content_lower:
                    try:
                        await message.delete()
                        await message.channel.send(
                            f"🚫 {message.author.mention}, that word is not allowed!",
                            delete_after=5
                        )
                        return
                    except:
                        pass

    if "poem-" in message.channel.name.lower() and len(content.split()) > 5:
        res = await asyncio.to_thread(
            get_groq_text, f"Rate this poem from 1-5 and give a short reason:\n{content}"
        )
        await safe_send(message.channel, f"✍️ **Poem Review:**\n{res}", reply_to=message)

    if "song-" in message.channel.name.lower() and message.attachments:
        res = await asyncio.to_thread(
            get_groq_text, "Give a short professional critique of a song submission."
        )
        await safe_send(message.channel, f"🎧 **Music Review:**\n{res}", reply_to=message)

    for user_id, words in highlights.items():
        if int(user_id) == message.author.id:
            continue

        for word in words:
            if word in content_lower:
                user = bot.get_user(int(user_id))
                if user:
                    try:
                        embed = discord.Embed(
                            title="📌 Highlight Triggered!",
                            description=f"The word **'{word}'** was mentioned in {message.channel.mention}.",
                            color=discord.Color.gold()
                        )
                        embed.add_field(name="Author", value=message.author.name, inline=True)
                        embed.add_field(name="Message", value=content, inline=False)
                        embed.add_field(
                            name="Jump to Message",
                            value=f"[Click Here]({message.jump_url})"
                        )

                        await user.send(embed=embed)
                    except discord.Forbidden:
                        pass
                        
    uid = message.author.id
    now = datetime.now(timezone.utc)

    # NEW: spam protection can be toggled off (server-wide or per-channel,
    # optionally temporarily) via the natural-language control channel --
    # see is_spam_protection_disabled() in that section below.
    if not is_spam_protection_disabled(message.guild.id, message.channel.id):
        data = spam_tracker[uid]

        if data["last_strike_time"] and (now - data["last_strike_time"]).total_seconds() > 43200:
            data["strikes"] = 0

        data["messages"].append((message.content.lower(), now))
        data["messages"] = [m for m in data["messages"] if (now - m[1]).total_seconds() <= 10]

        history = [m[0] for m in data["messages"]]
        if len(history) >= 4 and all(x == history[-1] for x in history[-4:]):
            data["strikes"] += 1
            data["last_strike_time"] = now
            data["messages"] = []
            
            s = data["strikes"]
            if s <= 2:
                await message.channel.send(f"⚠️ {message.author.mention}, don't repeat that! Warning {s}/3")
            elif s == 3:
                add_warning(
                    message.guild.id,
                    str(message.author.id),
                    "Repeated message spam (auto-detected by spam protection)",
                    bot.user.id,
                    f"{bot.user.name} (AutoMod)"
                )

                dm_sent = False
                try:
                    await message.author.send("🚨 Official Warning: You've been warned for spamming. One more and you get a 1-day timeout.")
                    dm_sent = True
                except Exception as e:
                    print(f"⚠️ Couldn't DM final spam warning to {message.author}: {e}")

                if dm_sent:
                    await message.channel.send(f"⚠️ {message.author.mention}, **Official Warning issued** (spamming). Check your DMs. Check `?warnings` to see it on your record.")
                else:
                    await message.channel.send(
                        f"⚠️ {message.author.mention}, **Official Warning issued** for spamming (Couldn't DM you -- "
                        f"your DMs might be closed). Use `?warnings` to see it on your record. One more spam message "
                        f"and you'll be timed out for 1 day."
                    )
            elif s >= 4:
                try:
                    await message.author.timeout(timedelta(days=1), reason="Spamming")
                    await message.channel.send(f"🔇 {message.author.mention} timed out for 1 day for spamming.")
                    data["strikes"] = 0
                except: await message.channel.send("❌ Permission error: Can't timeout user.")

    if not message.author.bot: 
        guild_id = str(message.guild.id)
        user_id = str(message.author.id)
        if guild_id not in message_counts: message_counts[guild_id] = {}
        message_counts[guild_id][user_id] = message_counts[guild_id].get(user_id, 0) + 1
        save_data(message_counts, MESSAGES_FILE)

        update_server_mood(message.guild.id, message)

    gid_str = str(message.guild.id)
    author_id_str = str(message.author.id)
    if gid_str in emoji_spam_targets and author_id_str in emoji_spam_targets[gid_str]:
        try:
            emoji_str = emoji_spam_targets[gid_str][author_id_str]
            await message.add_reaction(discord.PartialEmoji.from_str(emoji_str))
        except Exception as e:
            print(f"⚠️ Emoji auto-react failed: {e}")

    for mention in message.mentions:
        mention_afk_key = f"{message.guild.id}_{mention.id}"
        if mention_afk_key in afk_users:
            data = afk_users[mention_afk_key]
            reason = data['reason']
            try:
                afk_time = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
                delta = datetime.now(timezone.utc) - afk_time
                if delta.days > 0: time_ago = f"{delta.days} days ago"
                elif delta.seconds >= 3600: time_ago = f"{delta.seconds // 3600} hours ago"
                elif delta.seconds >= 60: time_ago = f"{delta.seconds // 60} minutes ago"
                else: time_ago = "just now"
            except ValueError:
                time_ago = "an unknown time ago" 
            
            await message.reply(
                f"😴 {mention.mention} is currently AFK ({time_ago}): **{reason}**",
                mention_author=True,
                allowed_mentions=discord.AllowedMentions(users=[message.author])
            )

    content = message.content.lower()
    if "ashley" in content:
        await message.channel.send("You wrote the name of the pookiest member 😉")
    elif "swastik" in content:
        await message.channel.send("ye kon bhadu bkl haryanvi gendu ka naam hai 😡")

    if "kala majdur" in content:
        try:
            await message.channel.send("kya kaam krna saheb", file=discord.File(KALA_MAJDUR_IMAGE_PATH))
        except FileNotFoundError:
            print(f"⚠️ {KALA_MAJDUR_IMAGE_PATH} not found -- upload it to the same folder as bot.py.")
        except Exception as e:
            print(f"⚠️ Failed to send kala majdur image: {e}")

    guild_triggers = triggers.get(str(message.guild.id), {})
    for phrase, trigger_data in guild_triggers.items():
        if phrase in content:
            # NEW: if this trigger is restricted to one channel (?triggerchannel),
            # skip it entirely when typed anywhere else.
            restrict_channel_id = trigger_data.get("restrict_channel_id")
            if restrict_channel_id and str(message.channel.id) != restrict_channel_id:
                continue
            try:
                text = trigger_data.get("text")
                image_path = trigger_data.get("image_path")

                # NEW: if this channel has a transfer redirect configured
                # (?triggertransfer), send the response there instead of
                # the channel the trigger was typed in.
                transfer_map = server_config.get(str(message.guild.id), {}).get("trigger_transfer", {})
                dest_channel_id = transfer_map.get(str(message.channel.id))
                target_channel = message.channel
                if dest_channel_id:
                    resolved = message.guild.get_channel(int(dest_channel_id))
                    if resolved:
                        target_channel = resolved

                if image_path and os.path.exists(image_path):
                    await target_channel.send(content=text, file=discord.File(image_path))
                elif text:
                    await target_channel.send(text)
            except Exception as e:
                print(f"⚠️ Failed to send trigger for '{phrase}': {e}")
            break

    await bot.process_commands(message)

# --------------------------------------------------------
# 📢 REMINDER COMMANDS
# --------------------------------------------------------

@bot.command()
async def create(ctx):
    """Trigger the selection menu"""
    await ctx.send("Select your creative mode:", view=CreateMenu())

@bot.command()
async def hpoem(ctx):
    if "Poem-" in ctx.channel.name:
        tid = ctx.channel.id
        hint_tracker[tid] = hint_tracker.get(tid, 0) + 1
        if hint_tracker[tid] <= 3:
            res = await asyncio.to_thread(get_groq_text, "Give a cryptic hint for a poem about nature.")
            await safe_send(ctx.channel, f"💡 *Hint {hint_tracker[tid]}/3:* {res}")
        else:
            await ctx.send("No more hints!")

@bot.command()
async def suggesth(ctx):
    if "Song-" in ctx.channel.name:
        res = await asyncio.to_thread(get_groq_text, "Suggest 5 great Hindi songs of different genres.")
        await safe_send(ctx.channel, f"🎧 *Hindi Recommendations:*\n{res}")

@bot.command()
async def suggeste(ctx):
    if "Song-" in ctx.channel.name:
        res = await asyncio.to_thread(get_groq_text, "Suggest 5 great English songs of different genres.")
        await safe_send(ctx.channel, f"🎸 *English Recommendations:*\n{res}")
        
@bot.command(help="[Admins only] Posts the interactive self-role menu -- dropdowns built from this server's configured categories.")
@commands.has_permissions(administrator=True)
async def setup_roles(ctx):
    categories = get_guild_role_categories(ctx.guild.id)
    non_empty = {name: roles for name, roles in categories.items() if roles}
    if not non_empty:
        return await ctx.send("❌ No role categories have any roles yet. Add some with `?addnewsetup <category> <role>`.")

    chunks = chunk_categories(non_empty, size=5)
    for i, chunk in enumerate(chunks, 1):
        embed = discord.Embed(
            title="🎭 Choose Your Roles" if i == 1 else f"🎭 Choose Your Roles (continued {i})",
            description="Pick from the dropdowns below -- your selections apply instantly and privately. Reopen a dropdown any time to change your mind.",
            color=0x5865F2
        )
        for cat_name, roles in chunk.items():
            embed.add_field(name=f"{category_emoji(cat_name)} {cat_name}", value=", ".join(roles), inline=False)
        if i == 1 and ctx.guild.icon:
            embed.set_thumbnail(url=ctx.guild.icon.url)
        embed.set_footer(text="Only you can see the confirmation when you pick something.")
        await ctx.send(embed=embed, view=DynamicRolePicker(chunk))

@bot.command(name="addnewcategory", usage="<category name>", help="[Admins only] Adds a brand-new category to the self-role menu. Run ?setup_roles again afterward to refresh the posted menu.")
@commands.has_permissions(administrator=True)
async def add_new_role_category(ctx, *, category: str):
    category = category.strip()
    if not category:
        return await ctx.send("❌ Give the category a name. Usage: `?addnewcategory <name>`")

    categories = get_guild_role_categories(ctx.guild.id)
    matched = next((c for c in categories if c.lower() == category.lower()), None)
    if matched:
        return await ctx.send(f"❌ A category called **{matched}** already exists.")

    categories[category] = []
    save_data(role_menu_config, ROLE_MENU_CONFIG_FILE)
    await ctx.send(f"✅ Added new category **{category}**. Use `?addnewsetup \"{category}\" <role name>` to add roles to it, then re-run `?setup_roles` to refresh the posted menu.")
    await send_mod_log(ctx.guild, "Role Menu Category Added", f"**Category:** {category}", ctx.author)

@bot.command(name="addnewsetup", usage='<category> <role name>', help="[Admins only] Adds a new self-assignable role to a category -- auto-creates the role (colourless) if it doesn't exist yet. Run ?setup_roles again afterward to refresh the posted menu.")
@commands.has_permissions(administrator=True)
async def add_new_role_setup(ctx, category: str, *, role_name: str):
    role_name = role_name.strip()
    categories = get_guild_role_categories(ctx.guild.id)

    # Case-insensitive category matching so mods don't have to get capitalization exactly right
    matched_category = next((c for c in categories if c.lower() == category.lower()), None)
    if not matched_category:
        available = ", ".join(categories.keys()) if categories else "none yet"
        return await ctx.send(f'❌ No category called "{category}". Available: {available}. Create it first with `?addnewcategory {category}`.')

    if role_name in categories[matched_category]:
        return await ctx.send(f"❌ **{role_name}** is already in the **{matched_category}** category.")
    if len(categories[matched_category]) >= 25:
        return await ctx.send(f"❌ **{matched_category}** already has the max 25 options Discord allows in one dropdown.")

    categories[matched_category].append(role_name)
    save_data(role_menu_config, ROLE_MENU_CONFIG_FILE)

    # Auto-creates the actual Discord role now (colourless), so it exists
    # even before anyone's picked it from the menu yet.
    existing_role = discord.utils.get(ctx.guild.roles, name=role_name)
    if not existing_role:
        try:
            await ctx.guild.create_role(name=role_name[:100], color=discord.Color.default(), reason=f"Auto-created via ?addnewsetup by {ctx.author}")
        except discord.Forbidden:
            await ctx.send(f"⚠️ Added **{role_name}** to the menu, but I couldn't auto-create the role (missing Manage Roles permission) -- it'll be created the first time someone picks it, if I have permission by then.")

    await ctx.send(f"✅ Added **{role_name}** to **{matched_category}**. Re-run `?setup_roles` to refresh the posted menu so it shows up.")
    await send_mod_log(ctx.guild, "Role Menu Option Added", f"**Category:** {matched_category}\n**Role:** {role_name}", ctx.author)

@bot.command()
async def rep(ctx, member: discord.Member):
    
    global reputation 
    if reputation is None:reputation = {}
    """Give a reputation point to a helpful user."""
    if member.id == ctx.author.id:
       return await ctx.send("You can't give yourself reputation! 💀")
    
    now = datetime.now()

    if ctx.author.id in rep_cooldowns:
        if now < rep_cooldowns[ctx.author.id] + timedelta(hours=1):
            return await ctx.send("⏳ You can only give rep once per hour!")

    mid = str(member.id)
    reputation[mid] = reputation.get(mid, 0) + 1
    rep_cooldowns[ctx.author.id] = now
    save_data(reputation, REPUTATION_FILE)
    await ctx.send(f"⭐ {ctx.author.mention} gave a rep point to {member.mention}! (Total: {reputation[mid]})")

@bot.command()
async def profile(ctx, member: discord.Member = None):
    global reputation,message_counts
    if reputation is None:reputation = {}
    if message_counts is None:message_counts = {}
        
    """View server profile, reputation, and message count."""
    member = member or ctx.author
    mid = str(member.id)
    gid = str(ctx.guild.id)
    
    msgs = message_counts.get(gid, {}).get(mid, 0)
    reps = reputation.get(mid, 0)

    embed = discord.Embed(title=f"User Profile: {member.name}", color=member.color)
    embed.add_field(name="💬 Messages", value=f"`{msgs}`", inline=True)
    embed.add_field(name="⭐ Reputation", value=f"`{reps}`", inline=True)

    badges = get_earned_badges(member, ctx.guild)
    if badges:
        embed.add_field(name=f"🏅 Badges ({len(badges)})", value=" ".join(b[0] for b in badges), inline=False)

    embed.set_thumbnail(url=member.display_avatar.url)
    await ctx.send(embed=embed)

def get_earned_badges(member: discord.Member, guild: discord.Guild) -> list[tuple[str, str]]:
    gid, uid = str(guild.id), str(member.id)
    earned = []

    msg_count = message_counts.get(gid, {}).get(uid, 0)
    reps = reputation.get(uid, 0)
    record = member_history.get(gid, {}).get(uid, {})
    join_count = record.get("join_count", 1)

    account_age_days = (datetime.now(timezone.utc) - member.created_at.replace(tzinfo=timezone.utc)).days
    if account_age_days <= 7:
        earned.append(("🐣", "Newcomer -- account created within the last week"))

    if record.get("first_joined"):
        first_joined_dt = datetime.fromisoformat(record["first_joined"])
        server_age_days = (datetime.now(timezone.utc) - first_joined_dt).days
        if server_age_days >= 365:
            earned.append(("🏛️", "Veteran -- in the server for 1+ year"))

    if msg_count >= 500:
        earned.append(("🗣️", "Motormouth -- 500+ messages this week"))
    elif msg_count >= 100:
        earned.append(("💬", "Chatterbox -- 100+ messages this week"))

    if reps >= 5:
        earned.append(("⭐", "Reputable -- 5+ reputation points"))

    if uid in get_guild_marriages(guild.id):
        earned.append(("💍", "Taken -- currently married"))
    if get_guild_marriage_stats(guild.id).get(uid, 0) >= 3:
        earned.append(("💔", "Serial Dater -- married 3+ times"))

    if uid in birthdays:
        earned.append(("🎂", "Birthday Registered"))

    if join_count >= 2:
        earned.append(("🔁", "Boomerang -- rejoined the server before"))

    if any(r.name == WINNER_ROLE_NAME for r in member.roles):
        earned.append(("👑", "Chatterbox Champion -- current weekly leaderboard winner"))

    return earned

# NEW: True Global Rarity -- counts how many UNIQUE people across every server
# this bot is in currently qualify for each badge. Iterating every member of
# every guild is too expensive to do on every ?badges call, so this is cached
# for a while; fine for a bot in a handful of small/medium servers, but if
# this bot ever ends up in many large servers this should move to a periodic
# background job instead of computing live.
_global_badge_cache = {"data": None, "computed_at": None}
GLOBAL_BADGE_CACHE_MINUTES = 15

async def get_global_badge_counts():
    now = datetime.now(timezone.utc)
    cached = _global_badge_cache["data"]
    computed_at = _global_badge_cache["computed_at"]
    if cached is not None and computed_at and (now - computed_at).total_seconds() < GLOBAL_BADGE_CACHE_MINUTES * 60:
        return cached

    holders = defaultdict(set)
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot:
                continue
            for emoji, desc in get_earned_badges(member, guild):
                holders[desc].add(member.id)

    counts = {desc: len(uids) for desc, uids in holders.items()}
    _global_badge_cache["data"] = counts
    _global_badge_cache["computed_at"] = now
    return counts

@bot.command(help="Shows all the achievement badges a user has earned.")
async def badges(ctx, member: discord.Member = None):
    member = member or ctx.author
    earned = get_earned_badges(member, ctx.guild)

    if not earned:
        return await ctx.send(f"📭 {member.display_name} hasn't earned any badges yet -- get chatting!")

    async with ctx.typing():
        global_counts = await get_global_badge_counts()

    lines = []
    for emoji, desc in earned:
        count = global_counts.get(desc)
        rarity_note = f" — *{count} of everyone who's ever used this bot has this*" if count else ""
        lines.append(f"{emoji}  **{desc}**{rarity_note}")
    description = "\n".join(lines)
    embed = discord.Embed(
        title=f"🏅 {member.display_name}'s Badges ({len(earned)})",
        description=description,
        color=discord.Color.gold()
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    embed.set_footer(text="Rarity counted across every server this bot is in")
    await ctx.send(embed=embed)

@bot.command()
@commands.has_permissions(manage_expressions=True)
async def steal(ctx, emoji: discord.PartialEmoji):
    """Steal an emoji from another server just by mentioning it!"""
    try:
        img_data = await emoji.read()
        
        new_emoji = await ctx.guild.create_custom_emoji(name=emoji.name, image=img_data)
        await ctx.send(f"✅ Successfully stolen! {new_emoji}")
        await send_mod_log(ctx.guild, "Emoji Stolen", f"**Name:** {emoji.name}", ctx.author)
    except Exception as e:
        await ctx.send(f"❌ Couldn't steal that: {e}")

@bot.command()
@commands.has_permissions(manage_guild=True)
async def restrict(ctx, word: str):
    global restricted_words
    if restricted_words is None:
        restricted_words = []
        
    word = word.lower()
    if word not in restricted_words:
        restricted_words.append(word)
        save_data(restricted_words, RESTRICTED_WORDS_FILE)
        await ctx.send(f"🚫 Restricted: **{word}**")
        await send_mod_log(ctx.guild, "Word Restricted", f"**Word:** {word}", ctx.author)
    else:
        await ctx.send("Word is already restricted.")
        
@bot.command()
@commands.has_permissions(manage_guild=True)
async def unrestrict(ctx, word: str):
    """Remove a word from the blacklist."""
    word = word.lower()
    if word in restricted_words:
        restricted_words.remove(word)
        save_data(restricted_words, RESTRICTED_WORDS_FILE)
        await ctx.send(f"✅ The word *'{word}'* has been unrestricted.")
        await send_mod_log(ctx.guild, "Word Unrestricted", f"**Word:** {word}", ctx.author)
    else:
        await ctx.send("That word isn't in the restriction list.")

@bot.command()
@commands.has_permissions(manage_messages=True)
async def restrictedlist(ctx):
    """Show all currently banned words to moderators."""
    if not restricted_words:
        return await ctx.send("No words are currently restricted.")
    words = ", ".join([f"{w}" for w in restricted_words])
    await ctx.send(f"*Restricted Words:* {words}")
    
@bot.command(name="hl")
@commands.has_permissions(manage_messages=True)
async def add_highlight(ctx, *, word: str):
    """Adds a word to your DM highlight list (Mods only)."""
    word = word.lower()
    uid = str(ctx.author.id)
    
    if uid not in highlights:
        highlights[uid] = []
    
    if word in highlights[uid]:
        return await ctx.send(f"❌ You already have '{word}' highlighted!")
    
    highlights[uid].append(word)
    save_data(highlights, HIGHLIGHTS_FILE)
    await ctx.send(f"✅ I'll DM you whenever someone mentions **'{word}'**!")

@bot.command(name="unhl")
@commands.has_permissions(manage_messages=True)
async def remove_highlight(ctx, *, word: str):
    """Removes a word from your list (Mods only)."""
    word = word.lower()
    uid = str(ctx.author.id)
    
    if uid in highlights and word in highlights[uid]:
        highlights[uid].remove(word)
        save_data(highlights, HIGHLIGHTS_FILE)
        await ctx.send(f"🗑️ Removed **'{word}'** from your highlights.")
    else:
        await ctx.send(f"❌ You don't have '{word}' highlighted.")

@bot.command(name="listhl")
@commands.has_permissions(manage_messages=True)
async def list_highlights(ctx):
    """Shows all words you have highlighted (Mods only)."""
    uid = str(ctx.author.id)
    
    if uid not in highlights or not highlights[uid]:
        return await ctx.send("📝 You don't have any highlight words set.")
    
    word_list = "\n".join([f"{i+1}. {word}" for i, word in enumerate(highlights[uid])])
    
    embed = discord.Embed(
        title="Your Highlighted Words",
        description=word_list,
        color=discord.Color.blue()
    )
    await ctx.send(embed=embed)
    
@bot.command()
@commands.has_permissions(manage_nicknames=True)
async def nick(ctx, member: discord.Member, *, new_nickname: str):
    """Changes the nickname of a member. Only for Moderators."""
    try:
        if len(new_nickname) > 32:
            return await ctx.send("❌ That nickname is too long! (Max 32 chars)")

        old_name = member.display_name
        await member.edit(nick=new_nickname)
        
        embed = discord.Embed(
            title="Nickname Updated ✅",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc)
        )
        embed.add_field(name="User", value=member.mention, inline=False)
        embed.add_field(name="Old Name", value=old_name, inline=True)
        embed.add_field(name="New Name", value=new_nickname, inline=True)
        embed.set_footer(text=f"Changed by {ctx.author.name}")
        
        await ctx.send(embed=embed)
        await send_mod_log(ctx.guild, "Nickname Changed", f"**User:** {member.mention}\n**Old:** {old_name}\n**New:** {new_nickname}", ctx.author)

    except discord.Forbidden:
        await ctx.send("❌ **Error:** I cannot change this user's name. They might have a higher role than me, or I'm missing the 'Manage Nicknames' permission.")
    except Exception as e:
        print(f"Nick Command Error: {e}")
        await ctx.send("❌ Something went wrong.")

@nick.error
async def nick_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send(f"❌ {ctx.author.mention}, you need the **Manage Nicknames** permission to use this command!")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ **Usage:** `?nick @user New Name`")

@bot.command()
async def reset(ctx):
    """Clears your private AI chat history."""
    if ctx.author.id in user_chats:
        del user_chats[ctx.author.id]
        await ctx.send("✅ Memory cleared! Let's start a fresh conversation.")
    else:
        await ctx.send("We don't have an active chat session to reset.")
        
@bot.command(name='remind', usage='"<Title>" <DD/MM/YYYY> <HH:MM>', help="Sets a public reminder that will be delivered via DM to you.")
async def remind_public(ctx, title: str, date_str: str, time_str: str):
    """Sets a public reminder that will be delivered via DM to the author."""
    await create_reminder(ctx, title, date_str, time_str, private=False, recipient_id=ctx.author.id)

@bot.command(name='remindpvt', usage='"<Title>" <DD/MM/YYYY> <HH:MM>', help="Sets a private reminder (confirmation hidden) delivered via DM to you.")
async def remind_private(ctx, title: str, date_str: str, time_str: str):
    """Sets a private reminder that will be delivered via DM (only you see the confirmation)."""
    await create_reminder(ctx, title, date_str, time_str, private=True, recipient_id=ctx.author.id)

@bot.command(name='senddm', usage='<User ID/@mention> "<Title>" <DD/MM/YYYY> <HH:MM>', help="Schedules a message to be sent via DM to a specific user.")
async def send_scheduled_dm(ctx, recipient: discord.User, title: str, date_str: str, time_str: str):
    """Schedules a message to be sent via DM to a specific user (can be outside the server)."""
    if recipient.bot:
        return await ctx.send("❌ Cannot send scheduled DMs to other bots.")
    
    await create_reminder(ctx, title, date_str, time_str, private=False, recipient_id=recipient.id)

# --------------------------------------------------------
# 🎂 BIRTHDAY COMMAND
# --------------------------------------------------------
@bot.command(name='addbday', usage='<day> <month> <year>', help="Register your birthday so the bot can celebrate it.")
async def add_birthday(ctx, day: int, month: int, year: int):
    """Registers the caller's birthday (day, month, year). Validates it's a real calendar date."""
    try:
        datetime(year=2000, month=month, day=day)
    except ValueError:
        return await ctx.send('❌ That\'s not a valid date. Usage: `?addbday <day> <month> <year>` e.g. `?addbday 15 8 2001`')

    current_year = datetime.now(IST_TIMEZONE).year
    if year < 1900 or year > current_year:
        return await ctx.send("❌ Please provide a realistic birth year.")

    birthdays[str(ctx.author.id)] = {"day": day, "month": month, "year": year}
    save_data(birthdays, BIRTHDAYS_FILE)

    await ctx.send(
        f"🎂 Got it, {ctx.author.mention}! Your birthday is registered as **{day:02d}/{month:02d}/{year}**. "
        f"I'll make sure to celebrate it when the day comes!"
    )

@bot.command(name='activatebday', help="[Mods only] Instantly runs the full birthday celebration for you, right now -- for testing/demo purposes.")
@commands.has_permissions(manage_guild=True)
async def activate_birthday_demo(ctx):
    """Mods-only: immediately triggers the real birthday celebration flow (announcement,
    role, DM) for the command author, without touching their registered birthday date
    or the daily birthday_check_loop logic in any way."""
    await ctx.send(
        f"🎬 Running the full birthday celebration for {ctx.author.mention} right now "
        f"(this is the real thing -- it'll ping @everyone, create a temporary role, and send a DM)..."
    )

    bday = birthdays.get(str(ctx.author.id))
    if not bday:
        now_ist = datetime.now(IST_TIMEZONE)
        bday = {"day": now_ist.day, "month": now_ist.month, "year": None}

    issues = await celebrate_birthday(ctx.guild, ctx.author, bday)

    if issues:
        await ctx.send(
            "⚠️ **Celebration ran, but a few things need attention:**\n" + "\n".join(issues)
        )
    else:
        await ctx.send("✅ Birthday celebration ran cleanly with no issues!")

# --------------------------------------------------------
# 📦 TIME CAPSULE
# --------------------------------------------------------
@bot.command(name="timecapsule", usage='<DD/MM/YYYY> <HH:MM> <message>', help="Seals a message that gets publicly revealed in this channel at a future date/time.")
async def timecapsule(ctx, date_str: str, time_str: str, *, message: str):
    try:
        deliver_at = parse_reminder_time(date_str, time_str)
    except ValueError as e:
        return await ctx.send(f"❌ Time Error: {e}")

    capsule_id = str(datetime.now().timestamp())
    timecapsules[capsule_id] = {
        "guild_id": ctx.guild.id,
        "channel_id": ctx.channel.id,
        "author_id": ctx.author.id,
        "message": message,
        "deliver_at": deliver_at.isoformat()
    }
    save_data(timecapsules, TIMECAPSULES_FILE)

    reveal_ist = deliver_at.astimezone(IST_TIMEZONE).strftime('%A, %d %B %Y at %I:%M %p IST')
    embed = discord.Embed(
        title="📦 Time Capsule Sealed!",
        description=f"Your message is locked away and will be revealed right here on **{reveal_ist}**.",
        color=discord.Color.dark_gold()
    )
    await ctx.send(embed=embed)


# --------------------------------------------------------
# 🏆 LEADERBOARD COMMAND 
# --------------------------------------------------------

@bot.command(aliases=['lb'], help="Displays the current top 10 message count rankings for the week.")
async def leaderboard(ctx):
    """Displays the current top 10 message count rankings for the week."""
    guild_id_str = str(ctx.guild.id)
    
    if guild_id_str not in message_counts or not message_counts[guild_id_str]:
        return await ctx.send("No messages have been tracked yet this week! Time to chat!")

    all_counts = message_counts[guild_id_str]
    sorted_list = sorted(all_counts.items(), key=itemgetter(1), reverse=True)
    
    leaderboard_text = ""
    trophies = {0: "🥇", 1: "🥈", 2: "🥉"} 
    
    for index, (user_id_str, count) in enumerate(sorted_list[:10]):
        member = ctx.guild.get_member(int(user_id_str))
        if member is None or member.bot: continue

        rank_display = trophies.get(index, f"#{index + 1}")
        leaderboard_text += f"{rank_display} **{member.display_name}**: `{count}` messages\n"

    now_ist = datetime.now(IST_TIMEZONE)
    days_until_sunday = (6 - now_ist.weekday() + 7) % 7
    if days_until_sunday == 0 and (now_ist.hour > 0 or now_ist.minute > 0): days_until_sunday = 7
    
    reset_date = (now_ist + timedelta(days=days_until_sunday)).replace(hour=0, minute=0, second=0, microsecond=0)
    time_remaining = reset_date - now_ist
    hours, remainder = divmod(time_remaining.total_seconds(), 3600)
    minutes, _ = divmod(remainder, 60)

    embed = discord.Embed(
        title="💬 Weekly Message Leaderboard 🏆",
        description="Top chatters of the week! Rankings reset every Sunday at 12:00 AM IST.",
        color=discord.Color.blue()
    )
    if leaderboard_text:
        embed.add_field(name="Current Top 10", value=leaderboard_text, inline=False)
    else:
         embed.add_field(name="Current Top 10", value="No valid users to display yet!", inline=False)
         
    embed.set_footer(text=f"Reset in: {int(hours)} hours and {int(minutes)} minutes.")
    await ctx.send(embed=embed)

@bot.command(name="resetlb", aliases=["resetleaderboard"], usage="@user", help="[Mods only] Resets a specific user's weekly leaderboard message count to zero.")
@commands.has_permissions(manage_messages=True)
async def reset_leaderboard_for_user(ctx, member: discord.Member):
    gid = str(ctx.guild.id)
    uid = str(member.id)
    old_count = message_counts.get(gid, {}).get(uid, 0)
    if old_count == 0:
        return await ctx.send(f"❌ {member.display_name} already has no messages counted this week.")

    message_counts.setdefault(gid, {})[uid] = 0
    save_data(message_counts, MESSAGES_FILE)
    record_undo(ctx.guild.id, "resetlb", ctx.author.id, {"user_id": member.id, "old_count": old_count})

    await ctx.send(f"✅ Reset {member.mention}'s weekly leaderboard count from **{old_count}** to **0**. Use `?undo` to reverse this.")
    await send_mod_log(ctx.guild, "Leaderboard Reset (User)", f"**User:** {member.mention}\n**Old Count:** {old_count}", ctx.author)


# --------------------------------------------------------
# 🛡️ MODERATION COMMANDS
# --------------------------------------------------------

def add_warning(guild_id: int, user_id_str: str, reason: str, moderator_id: int, moderator_name: str):
    """Shared helper for creating a real, official warning entry -- used by both
    the ?warn command and the automated spam-protection system, so spam
    warnings show up properly in ?warnings just like a mod-issued one would.
    Scoped per-server -- a warning on one server doesn't show up on another."""
    gid_str = str(guild_id)
    warnings_data.setdefault(gid_str, {})
    if user_id_str not in warnings_data[gid_str]:
        warnings_data[gid_str][user_id_str] = []

    warnings_data[gid_str][user_id_str].append({
        "reason": reason,
        "moderator_id": moderator_id,
        "moderator_name": moderator_name,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    })
    save_data(warnings_data, WARNINGS_FILE)

@bot.command()
@commands.has_permissions(kick_members=True)
async def warn(ctx, member: discord.Member, *, reason="No reason provided"):
    add_warning(ctx.guild.id, str(member.id), reason, ctx.author.id, ctx.author.name)
    record_undo(ctx.guild.id, "warn", ctx.author.id, {"user_id": member.id})

    # BUG FIX: this used to only post in the channel and never actually DM the
    # warned user -- the message just said "has been warned" without any DM
    # ever being sent. Now it actually attempts the DM, with a real embed,
    # and the channel message reflects whether it landed.
    total_warnings = len(warnings_data.get(str(ctx.guild.id), {}).get(str(member.id), []))
    dm_embed = discord.Embed(
        title="⚠️ You've received a warning",
        description=f"You were warned in **{ctx.guild.name}**.",
        color=discord.Color.orange(),
        timestamp=datetime.now(timezone.utc)
    )
    dm_embed.add_field(name="Reason", value=reason, inline=False)
    dm_embed.add_field(name="Moderator", value=ctx.author.name, inline=True)
    dm_embed.add_field(name="Total Warnings", value=str(total_warnings), inline=True)

    dm_sent = False
    try:
        await member.send(embed=dm_embed)
        dm_sent = True
    except Exception as e:
        print(f"⚠️ Couldn't DM warning to {member}: {e}")

    if dm_sent:
        await ctx.send(f"⚠️ **{member.mention}** has been warned for: {reason} (DM sent)")
    else:
        await ctx.send(f"⚠️ **{member.mention}** has been warned for: {reason} (⚠️ couldn't DM them -- their DMs are likely closed)")

    await send_mod_log(ctx.guild, "Member Warned",
                       f"**User:** {member.mention} ({member.id})\n**Reason:** {reason}\n**Total Warnings:** {total_warnings}",
                       ctx.author)

# --------------------------------------------------------
# 🗳️ MODERATION POLLS (Mods only)
# --------------------------------------------------------
MODPOLL_MAX_WAIT_SECONDS = 300  # poll expires if the vote target is never reached within 5 minutes
MODPOLL_CHECK_INTERVAL = 5

@bot.command(name="modpoll", usage="@user <votes_needed> <kick|warn|ban|timeout> [duration_minutes if timeout] [reason]",
             help="[Mods only] Starts a vote -- as soon as it hits the target vote count, the bot immediately carries out the action.")
async def mod_poll(ctx, member: discord.Member, votes_needed: int, action: str, *, rest: str = ""):
    action = action.lower()
    if action not in ("kick", "warn", "ban", "timeout"):
        return await ctx.send('❌ Action must be one of: `kick`, `warn`, `ban`, `timeout`. Usage: `?modpoll @user <votes> <action> [reason]`')

    if votes_needed < 1 or votes_needed > 100:
        return await ctx.send("❌ Votes needed must be between 1 and 100.")

    required_perm = "ban_members" if action == "ban" else ("moderate_members" if action == "timeout" else "kick_members")
    if not getattr(ctx.author.guild_permissions, required_perm):
        return await ctx.send(f"❌ You need the **{required_perm.replace('_', ' ').title()}** permission to start a `{action}` poll.")

    if member.id == ctx.author.id:
        return await ctx.send("❌ You can't poll a moderation action against yourself.")
    if member.bot:
        return await ctx.send("❌ Can't run a moderation poll on a bot.")
    if ctx.guild.me.top_role <= member.top_role:
        return await ctx.send("❌ I can't act on this member -- their role is positioned too high for me.")

    # NEW: timeout needs its own duration, parsed as the first word of `rest`.
    duration_minutes = None
    reason = rest.strip()
    if action == "timeout":
        parts = rest.split(maxsplit=1)
        if not parts or not parts[0].isdigit():
            return await ctx.send('❌ Timeout polls need a duration in minutes. Usage: `?modpoll @user <votes> timeout <minutes> [reason]`')
        duration_minutes = int(parts[0])
        if duration_minutes < 1 or duration_minutes > 40320:  # 28 days, Discord's own timeout cap
            return await ctx.send("❌ Timeout duration must be between 1 minute and 28 days (40320 minutes).")
        reason = parts[1].strip() if len(parts) > 1 else "No reason provided"
    elif not reason:
        reason = "No reason provided"

    action_display = f"timed out for {duration_minutes} minutes" if action == "timeout" else f"{action}ed"

    embed = discord.Embed(
        title=f"🗳️ Moderation Poll: {action.upper()}",
        description=(
            f"Should **{member.mention}** be **{action_display}**?\n\n**Reason:** {reason}\n\n"
            f"React 👍 -- once **{votes_needed}** people vote yes, this triggers immediately.\n"
            f"Poll expires in {MODPOLL_MAX_WAIT_SECONDS // 60} minutes if the target isn't reached."
        ),
        color=discord.Color.orange()
    )
    embed.set_footer(text=f"Started by {ctx.author.display_name} -- their own vote doesn't count")
    msg = await ctx.send(embed=embed)
    await msg.add_reaction("👍")
    await msg.add_reaction("👎")

    elapsed = 0
    triggered = False
    yes_votes = no_votes = 0

    while elapsed < MODPOLL_MAX_WAIT_SECONDS:
        await asyncio.sleep(MODPOLL_CHECK_INTERVAL)
        elapsed += MODPOLL_CHECK_INTERVAL
        try:
            msg = await ctx.channel.fetch_message(msg.id)
        except Exception:
            return

        # Pulls the actual reactor list (not just the raw count) so the bot's
        # own reaction and the poll-starter's own vote don't count -- a mod
        # shouldn't be able to just self-vote and pass their own poll.
        yes_users, no_users = set(), set()
        for reaction in msg.reactions:
            if str(reaction.emoji) == "👍":
                async for user in reaction.users():
                    if not user.bot and user.id != ctx.author.id:
                        yes_users.add(user.id)
            elif str(reaction.emoji) == "👎":
                async for user in reaction.users():
                    if not user.bot and user.id != ctx.author.id:
                        no_users.add(user.id)
        yes_votes, no_votes = len(yes_users), len(no_users)

        if yes_votes >= votes_needed:
            triggered = True
            break

    if not triggered:
        await ctx.send(f"🗳️ **Poll expired** -- only reached {yes_votes}/{votes_needed} needed votes. No action taken on {member.mention}.")
        await send_mod_log(ctx.guild, f"Moderation Poll Expired ({action})",
                           f"**Target:** {member.mention}\n**Votes:** {yes_votes}/{votes_needed}\n**Started by:** {ctx.author.mention}", ctx.author)
        return

    try:
        if action == "kick":
            await member.kick(reason=f"Moderation poll passed ({yes_votes}/{votes_needed} votes). Reason: {reason}")
            await ctx.send(f"✅ **Poll passed** ({yes_votes}/{votes_needed} votes) -- {member.mention} has been kicked.")
        elif action == "ban":
            await ctx.guild.ban(member, reason=f"Moderation poll passed ({yes_votes}/{votes_needed} votes). Reason: {reason}")
            await ctx.send(f"✅ **Poll passed** ({yes_votes}/{votes_needed} votes) -- {member.mention} has been banned.")
        elif action == "timeout":
            await member.timeout(discord.utils.utcnow() + timedelta(minutes=duration_minutes),
                                 reason=f"Moderation poll passed ({yes_votes}/{votes_needed} votes). Reason: {reason}")
            await ctx.send(f"✅ **Poll passed** ({yes_votes}/{votes_needed} votes) -- {member.mention} has been timed out for {duration_minutes} minutes.")
        elif action == "warn":
            add_warning(ctx.guild.id, str(member.id), f"{reason} (via moderation poll, {yes_votes}/{votes_needed} votes)", ctx.author.id, ctx.author.name)
            try:
                dm_embed = discord.Embed(
                    title="⚠️ You've received a warning",
                    description=f"You were warned in **{ctx.guild.name}** via a community moderation poll.",
                    color=discord.Color.orange()
                )
                dm_embed.add_field(name="Reason", value=reason, inline=False)
                await member.send(embed=dm_embed)
            except Exception:
                pass
            await ctx.send(f"✅ **Poll passed** ({yes_votes}/{votes_needed} votes) -- {member.mention} has been warned.")

        await send_mod_log(ctx.guild, f"Moderation Poll Passed ({action})",
                           f"**Target:** {member.mention}\n**Votes:** {yes_votes}/{votes_needed}\n**Reason:** {reason}\n**Started by:** {ctx.author.mention}", ctx.author)
    except discord.Forbidden:
        await ctx.send(f"❌ Poll passed, but I don't have permission to {action} this member.")

# --------------------------------------------------------
# 😂 EMOJI AUTO-REACT (Mods only)
# --------------------------------------------------------
@bot.command(name="spamem", usage="@user <emoji>", help="[Mods only] Bot auto-reacts with that emoji on every message the user sends, until stopped.")
@commands.has_permissions(manage_messages=True)
@commands.check(not_blocked_from_trigger_commands)
async def spam_emoji(ctx, member: discord.Member, emoji: str):
    try:
        parsed = discord.PartialEmoji.from_str(emoji)
        await ctx.message.add_reaction(parsed)
    except Exception:
        return await ctx.send("❌ That doesn't look like a valid emoji I can use. Try a default emoji or one from a server I'm in.")

    gid, uid = str(ctx.guild.id), str(member.id)
    emoji_spam_targets.setdefault(gid, {})[uid] = emoji
    save_data(emoji_spam_targets, EMOJI_SPAM_FILE)
    await ctx.send(f"✅ I'll now react with {emoji} on every message {member.mention} sends. Use `?stopspamem @user` to stop.")
    await send_mod_log(ctx.guild, "Emoji Auto-React Set", f"**Target:** {member.mention}\n**Emoji:** {emoji}", ctx.author)

@bot.command(name="stopspamem", usage="@user", help="[Mods only] Stops the emoji auto-react for a user.")
@commands.has_permissions(manage_messages=True)
@commands.check(not_blocked_from_trigger_commands)
async def stop_spam_emoji(ctx, member: discord.Member):
    gid, uid = str(ctx.guild.id), str(member.id)
    if gid in emoji_spam_targets and uid in emoji_spam_targets[gid]:
        del emoji_spam_targets[gid][uid]
        save_data(emoji_spam_targets, EMOJI_SPAM_FILE)
        await ctx.send(f"✅ Stopped auto-reacting to {member.mention}'s messages.")
        await send_mod_log(ctx.guild, "Emoji Auto-React Stopped", f"**Target:** {member.mention}", ctx.author)
    else:
        await ctx.send(f"❌ {member.display_name} isn't currently being auto-reacted to.")

# --------------------------------------------------------
# 🎯 CUSTOM TRIGGERS (Mods only)
# --------------------------------------------------------
@bot.command(name="addtrigger", usage='"<phrase>" [text] (attach an image too if you want)', help="[Mods only] Makes the bot auto-reply with text and/or an image whenever someone says <phrase>.")
@commands.has_permissions(manage_messages=True)
@commands.check(not_blocked_from_trigger_commands)
async def add_trigger(ctx, phrase: str, *, text: str = None):
    phrase = phrase.lower().strip()
    if not phrase:
        return await ctx.send('❌ Usage: `?addtrigger "phrase" [response text]` (attach an image too if you want).')

    image_path = None
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        if not (attachment.content_type and attachment.content_type.startswith("image/")):
            return await ctx.send("❌ That attachment isn't an image.")
        ext = os.path.splitext(attachment.filename)[1] or ".png"
        safe_name = f"{ctx.guild.id}_{hashlib.sha256(phrase.encode()).hexdigest()[:16]}{ext}"
        image_path = os.path.join(TRIGGER_IMAGES_DIR, safe_name)
        await attachment.save(image_path)

    if not text and not image_path:
        return await ctx.send('❌ Provide response text, an image attachment, or both. Usage: `?addtrigger "phrase" [text]`')

    gid = str(ctx.guild.id)
    # NEW: preserve an existing channel restriction (?triggerchannel) if this
    # phrase already had one and is just being updated with new text/image.
    existing = triggers.get(gid, {}).get(phrase, {})
    triggers.setdefault(gid, {})[phrase] = {"text": text, "image_path": image_path, "restrict_channel_id": existing.get("restrict_channel_id")}
    save_data(triggers, TRIGGERS_FILE)

    await ctx.send(f"✅ Trigger added: whenever someone says **\"{phrase}\"**, I'll reply with {'text + an image' if text and image_path else 'text' if text else 'an image'}.")
    await send_mod_log(ctx.guild, "Trigger Added", f"**Phrase:** {phrase}", ctx.author)

@bot.command(name="removetrigger", usage='"<phrase>"', help="[Mods only] Removes a custom trigger.")
@commands.has_permissions(manage_messages=True)
@commands.check(not_blocked_from_trigger_commands)
async def remove_trigger(ctx, *, phrase: str):
    phrase = phrase.lower().strip()
    gid = str(ctx.guild.id)

    if gid not in triggers or phrase not in triggers[gid]:
        return await ctx.send(f'❌ No trigger found for "{phrase}".')

    image_path = triggers[gid][phrase].get("image_path")
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception as e:
            print(f"⚠️ Failed to delete trigger image file: {e}")

    del triggers[gid][phrase]
    save_data(triggers, TRIGGERS_FILE)
    await ctx.send(f'✅ Removed trigger for "{phrase}".')
    await send_mod_log(ctx.guild, "Trigger Removed", f"**Phrase:** {phrase}", ctx.author)

@bot.command(name="triggerchannel", usage='"<phrase>" [#channel]', help="[Mods only] Restricts a trigger to only fire in one channel. Omit the channel to clear the restriction.")
@commands.has_permissions(manage_messages=True)
@commands.check(not_blocked_from_trigger_commands)
async def trigger_channel_restrict(ctx, phrase: str, channel: discord.TextChannel = None):
    phrase = phrase.lower().strip()
    gid = str(ctx.guild.id)

    if gid not in triggers or phrase not in triggers[gid]:
        return await ctx.send(f'❌ No trigger found for "{phrase}". Use `?addtrigger` first.')

    triggers[gid][phrase]["restrict_channel_id"] = str(channel.id) if channel else None
    save_data(triggers, TRIGGERS_FILE)

    if channel:
        await ctx.send(f'✅ **"{phrase}"** will now only fire in {channel.mention}.')
        await send_mod_log(ctx.guild, "Trigger Channel-Restricted", f"**Phrase:** {phrase}\n**Channel:** {channel.mention}", ctx.author)
    else:
        await ctx.send(f'✅ **"{phrase}"** can now fire in any channel again.')
        await send_mod_log(ctx.guild, "Trigger Channel Restriction Cleared", f"**Phrase:** {phrase}", ctx.author)

@bot.command(name="triggertransfer", usage="#source #destination", help="[Mods only] Any trigger typed in #source posts its response in #destination instead.")
@commands.has_permissions(manage_messages=True)
@commands.check(not_blocked_from_trigger_commands)
async def trigger_transfer(ctx, source: discord.TextChannel, destination: discord.TextChannel):
    gid = str(ctx.guild.id)
    server_config.setdefault(gid, {}).setdefault("trigger_transfer", {})
    server_config[gid]["trigger_transfer"][str(source.id)] = str(destination.id)
    save_data(server_config, SERVER_CONFIG_FILE)

    await ctx.send(f"✅ Triggers typed in {source.mention} will now post their response in {destination.mention} instead.")
    await send_mod_log(ctx.guild, "Trigger Transfer Set", f"**Source:** {source.mention}\n**Destination:** {destination.mention}", ctx.author)

@bot.command(name="removetriggertransfer", usage="#source", help="[Mods only] Removes the transfer redirect for a source channel, so triggers there respond normally again.")
@commands.has_permissions(manage_messages=True)
@commands.check(not_blocked_from_trigger_commands)
async def remove_trigger_transfer(ctx, source: discord.TextChannel):
    gid = str(ctx.guild.id)
    transfer_map = server_config.get(gid, {}).get("trigger_transfer", {})
    if str(source.id) not in transfer_map:
        return await ctx.send(f"❌ {source.mention} doesn't have a transfer redirect set.")

    del transfer_map[str(source.id)]
    save_data(server_config, SERVER_CONFIG_FILE)
    await ctx.send(f"✅ Removed the transfer redirect for {source.mention}.")
    await send_mod_log(ctx.guild, "Trigger Transfer Removed", f"**Source:** {source.mention}", ctx.author)

@bot.command(name="listtriggers", help="Shows all custom triggers set up on this server.")
async def list_triggers(ctx):
    gid = str(ctx.guild.id)
    guild_triggers = triggers.get(gid, {})

    if not guild_triggers:
        return await ctx.send("📭 No custom triggers set up on this server yet. Mods can add one with `?addtrigger`.")

    lines = []
    for phrase, data in guild_triggers.items():
        kind = "text + image" if data.get("text") and data.get("image_path") else "text" if data.get("text") else "image"
        lines.append(f"• **{phrase}** ({kind})")

    embed = discord.Embed(title="🎯 Custom Triggers", description="\n".join(lines), color=discord.Color.blue())
    await ctx.send(embed=embed)

@bot.command()
@commands.has_permissions(moderate_members=True)
@commands.bot_has_permissions(moderate_members=True)
async def timeout(ctx, member: discord.Member, duration_minutes: int, *, reason: str = "No reason provided"):
    if ctx.author.top_role <= member.top_role and ctx.guild.owner_id != ctx.author.id:
        await ctx.send("You cannot timeout someone with a higher or equal role.")
        return
        
    duration = timedelta(minutes=duration_minutes)
    if duration.total_seconds() < 60 or duration.days > 28:
        await ctx.send("Timeout duration must be between 1 minute and 28 days.")
        return

    await member.timeout(discord.utils.utcnow() + duration, reason=reason)
    record_undo(ctx.guild.id, "timeout", ctx.author.id, {"user_id": member.id})
    await ctx.send(f"✅ {member.mention} has been timed out for **{duration_minutes} minutes**.")
    await send_mod_log(ctx.guild, "Member Timed Out", 
                       f"**User:** {member.mention}\n**Duration:** {duration_minutes} minutes\n**Reason:** {reason}", 
                       ctx.author)

@bot.command()
@commands.has_permissions(ban_members=True)
@commands.bot_has_permissions(ban_members=True)
async def ban(ctx, user: discord.User, *, reason: str = "No reason provided"):
    member = ctx.guild.get_member(user.id)
    if member and ctx.author.top_role <= member.top_role and ctx.guild.owner_id != ctx.author.id:
        await ctx.send("You cannot ban someone with a higher or equal role.")
        return
        
    try:
        await ctx.guild.ban(user, reason=reason)
        record_undo(ctx.guild.id, "ban", ctx.author.id, {"user_id": user.id})
        await ctx.send(f"🔨 {user.mention} has been banned.")
        await send_mod_log(ctx.guild, "Member Banned", 
                           f"**User:** {user.mention} ({user.id})\n**Reason:** {reason}", 
                           ctx.author)
    except discord.Forbidden:
        await ctx.send("I do not have permission to ban that member.")

@bot.command()
@commands.has_permissions(ban_members=True)
@commands.bot_has_permissions(ban_members=True)
async def unban(ctx, user_id_or_name: str):
    try:
        if user_id_or_name.isdigit():
            user_id = int(user_id_or_name)
            user = discord.Object(id=user_id)
            await ctx.guild.unban(user)
            await ctx.send(f"✅ User with ID `{user_id}` has been unbanned.")
            await send_mod_log(ctx.guild, "Member Unbanned", 
                               f"**User ID:** {user_id}\nUnbanned by: {ctx.author.mention}", 
                               ctx.author)
            return
            
        bans = [entry async for entry in ctx.guild.bans()]
        for ban_entry in bans:
            user = ban_entry.user
            if user.name.lower() == user_id_or_name.lower() or str(user).lower() == user_id_or_name.lower():
                await ctx.guild.unban(user)
                await ctx.send(f"✅ {user} has been unbanned.")
                await send_mod_log(ctx.guild, "Member Unbanned", 
                                   f"**User:** {user.name} ({user.id})\nUnbanned by: {ctx.author.mention}", 
                                   ctx.author)
                return
                
        await ctx.send("User not found in ban list (check ID or full tag).")
        
    except discord.Forbidden:
        await ctx.send("I do not have permission to unban this user.")
    except discord.NotFound:
        await ctx.send("This user is not banned.")

@bot.command()
@commands.has_permissions(kick_members=True)
@commands.bot_has_permissions(kick_members=True)
async def kick(ctx, member: discord.Member, *, reason="No reason provided"):
    if ctx.author.top_role <= member.top_role and ctx.guild.owner_id != ctx.author.id:
        await ctx.send("You can't kick someone with a higher or equal role!")
        return
    role_ids = [r.id for r in member.roles if r.name != "@everyone"]  # snapshotted for ?undo -- see undo_kick
    try:
        await member.kick(reason=reason)
        record_undo(ctx.guild.id, "kick", ctx.author.id, {"user_id": member.id, "role_ids": role_ids})
        await ctx.send(f"👢 {member.mention} has been kicked. Reason: {reason}")
        await send_mod_log(ctx.guild, "Member Kicked", 
                           f"**User:** {member.mention}\n**Reason:** {reason}", 
                           ctx.author)
    except discord.Forbidden:
        await ctx.send("I do not have permission to kick this member.")

@bot.command(help="Deletes a specified number of messages (max 100).")
@commands.has_permissions(manage_messages=True)
async def purge(ctx, amount: int):
    if amount < 1 or amount > 100:
        return await ctx.send("❌ Please specify an amount between 1 and 100.")
    
    try:
        deleted = await ctx.channel.purge(limit=amount + 1)
        record_undo(ctx.guild.id, "purge", ctx.author.id, {
            "channel_id": ctx.channel.id,
            "messages": [
                {"author_name": m.author.display_name, "content": m.content, "is_bot": m.author.bot}
                for m in reversed(deleted) if m.id != ctx.message.id
            ],
        })
        await ctx.send(f"🗑️ Deleted **{len(deleted) - 1}** messages.", delete_after=5)
        await send_mod_log(ctx.guild, "Messages Purged", f"Channel: {ctx.channel.mention}\nAmount: {len(deleted) - 1}\nModerator: {ctx.author.mention}", ctx.author)
    except discord.Forbidden:
        await ctx.send("❌ I do not have permission to delete messages.")
    except Exception as e:
        await ctx.send(f"❌ An error occurred: {e}")
        
@bot.command(help="Prevents the @everyone role from sending messages.")
@commands.has_permissions(manage_channels=True)
async def lock(ctx):
    await ctx.channel.set_permissions(ctx.guild.default_role, send_messages=False)
    record_undo(ctx.guild.id, "lock_channel", ctx.author.id, {"channel_id": ctx.channel.id})
    await ctx.send("🔒 Channel locked.")
    await send_mod_log(ctx.guild, "Channel Locked", f"Channel: {ctx.channel.mention}", ctx.author)

@bot.command(help="Allows the @everyone role to send messages.")
@commands.has_permissions(manage_channels=True)
async def unlock(ctx):
    await ctx.channel.set_permissions(ctx.guild.default_role, send_messages=True)
    record_undo(ctx.guild.id, "unlock_channel", ctx.author.id, {"channel_id": ctx.channel.id})
    await ctx.send("🔓 Channel unlocked.")
    await send_mod_log(ctx.guild, "Channel Unlocked", f"Channel: {ctx.channel.mention}", ctx.author)

# --------------------------------------------------------
# 🔒 JAIL SYSTEM (?jail / ?unjail)
# --------------------------------------------------------
# A jailed member gets a "🔒 Jailed" role that's denied view access on every
# channel EXCEPT the one configured with ?setjailchannel -- so they can't see
# or type anywhere else (text or voice) on the server, only that one channel.
# The bot only manages the Jailed role's own access; it never touches other
# roles/members' permissions on the jail channel, so set that channel up
# yourself (hidden from @everyone, visible to your mod role) the same way
# you would any other mod-only channel -- the bot's overwrite just adds the
# Jailed role on top of whatever you've already configured there.
JAIL_ROLE_NAME = "🔒 Jailed"

async def get_or_create_jail_role(guild: discord.Guild) -> Optional[discord.Role]:
    role = discord.utils.get(guild.roles, name=JAIL_ROLE_NAME)
    if role:
        return role
    try:
        return await guild.create_role(
            name=JAIL_ROLE_NAME, color=discord.Color.dark_grey(),
            permissions=discord.Permissions.none(),
            reason="Auto-created for the jail system (?jail/?unjail)"
        )
    except discord.Forbidden:
        return None

async def apply_jail_overwrites(guild: discord.Guild, jail_role: discord.Role, jail_channel_id: str):
    """Loops every channel in the server and denies the Jailed role view/send/
    connect access, except the configured jail channel where it's explicitly
    granted view+send. Run once from ?setjailchannel (and again automatically
    whenever a new channel is created, via on_guild_channel_create below) --
    NOT on every single ?jail, since looping every channel per-jail would be
    needlessly slow/rate-limit-risky on larger servers."""
    for channel in guild.channels:
        try:
            if str(channel.id) == jail_channel_id:
                await channel.set_permissions(jail_role, view_channel=True, send_messages=True, read_message_history=True, reason="Jail system setup")
            else:
                await channel.set_permissions(jail_role, view_channel=False, send_messages=False, connect=False, reason="Jail system setup")
        except discord.Forbidden:
            print(f"⚠️ Missing permission to set jail overwrites on channel {channel.name} ({channel.id}) in {guild.name}.")
        except Exception as e:
            print(f"⚠️ Failed to set jail overwrite on channel {channel.id}: {e}")

@bot.command(name="setjailchannel", usage="#channel", help="[Mods only] Sets the jail channel and locks the Jailed role out of every other channel (text + voice) on the server.")
@commands.has_permissions(manage_guild=True)
@commands.bot_has_permissions(manage_roles=True, manage_channels=True)
async def set_jail_channel(ctx, channel: discord.TextChannel):
    gid = str(ctx.guild.id)
    server_config.setdefault(gid, {})["jail_channel_id"] = str(channel.id)
    save_data(server_config, SERVER_CONFIG_FILE)

    role = await get_or_create_jail_role(ctx.guild)
    if not role:
        return await ctx.send("❌ Couldn't create the Jailed role -- I need **Manage Roles** permission.")

    await ctx.send(f"⚙️ Setting up jail permissions across every channel for {channel.mention} -- this may take a moment on larger servers...")
    async with ctx.typing():
        await apply_jail_overwrites(ctx.guild, role, str(channel.id))

    await ctx.send(
        f"✅ Jail channel set to {channel.mention}. Anyone with the **{JAIL_ROLE_NAME}** role can now only "
        f"see/type there -- everywhere else (including voice channels) is hidden from them.\n"
        f"⚠️ I only manage the Jailed role's own access -- make sure {channel.mention}'s other permissions "
        f"(hidden from regular members, visible to your mod role) are set up the way you want, the same as "
        f"any other mod-only channel."
    )
    await send_mod_log(ctx.guild, "Jail Channel Configured", f"**Channel:** {channel.mention}", ctx.author)

@bot.command(name="jail", usage="@user [reason]", help="[Mods only] Jails a member -- they can only see/type in the configured jail channel until ?unjail.")
@commands.has_permissions(moderate_members=True)
@commands.bot_has_permissions(manage_roles=True)
async def jail_member(ctx, member: discord.Member, *, reason: str = "No reason provided"):
    gid = str(ctx.guild.id)
    jail_channel_id = server_config.get(gid, {}).get("jail_channel_id")
    if not jail_channel_id:
        return await ctx.send("❌ No jail channel configured yet. A mod needs to run `?setjailchannel #channel` first.")
    if member.id == ctx.author.id:
        return await ctx.send("❌ You can't jail yourself!")
    if member.bot:
        return await ctx.send("❌ Can't jail a bot.")
    if ctx.author.top_role <= member.top_role and ctx.guild.owner_id != ctx.author.id:
        return await ctx.send("❌ You can't jail someone with a higher or equal role.")

    role = discord.utils.get(ctx.guild.roles, name=JAIL_ROLE_NAME)
    if not role:
        return await ctx.send("❌ The Jailed role doesn't exist yet -- a mod needs to run `?setjailchannel #channel` first.")
    if role in member.roles:
        return await ctx.send(f"❌ {member.display_name} is already jailed.")

    try:
        await member.add_roles(role, reason=f"Jailed by {ctx.author}: {reason}")
    except discord.Forbidden:
        return await ctx.send("❌ I can't assign that role -- my role needs to be positioned above the Jailed role and above this member's highest role.")

    record_undo(ctx.guild.id, "jail", ctx.author.id, {"user_id": member.id})

    if member.voice and member.voice.channel:
        try:
            await member.move_to(None, reason="Jailed -- removed from voice")
        except Exception:
            pass

    jail_channel = ctx.guild.get_channel(int(jail_channel_id))
    embed = discord.Embed(
        title="🔒 Member Jailed",
        description=f"{member.mention} has been jailed" + (f" -- see {jail_channel.mention}" if jail_channel else "") + ".",
        color=discord.Color.dark_grey()
    )
    embed.add_field(name="Reason", value=reason, inline=False)
    await ctx.send(embed=embed)

    try:
        await member.send(f"🔒 You've been jailed in **{ctx.guild.name}**. Reason: {reason}\nYou can only see/type in the jail channel until a moderator unjails you.")
    except Exception:
        pass

    await send_mod_log(ctx.guild, "Member Jailed", f"**User:** {member.mention} ({member.id})\n**Reason:** {reason}", ctx.author)

@bot.command(name="unjail", usage="@user", help="[Mods only] Releases a jailed member.")
@commands.has_permissions(moderate_members=True)
@commands.bot_has_permissions(manage_roles=True)
async def unjail_member(ctx, member: discord.Member):
    role = discord.utils.get(ctx.guild.roles, name=JAIL_ROLE_NAME)
    if not role or role not in member.roles:
        return await ctx.send(f"❌ {member.display_name} isn't currently jailed.")
    try:
        await member.remove_roles(role, reason=f"Unjailed by {ctx.author}")
    except discord.Forbidden:
        return await ctx.send("❌ I don't have permission to remove that role.")

    await ctx.send(f"🔓 {member.mention} has been released from jail.")
    try:
        await member.send(f"🔓 You've been released from jail in **{ctx.guild.name}**.")
    except Exception:
        pass
    await send_mod_log(ctx.guild, "Member Unjailed", f"**User:** {member.mention} ({member.id})", ctx.author)

# NEW: keeps newly-created channels locked for the Jailed role automatically,
# so a mod doesn't have to re-run ?setjailchannel every time the server adds
# a channel.
@bot.event
async def on_guild_channel_create(channel):
    guild = channel.guild
    jail_channel_id = server_config.get(str(guild.id), {}).get("jail_channel_id")
    if not jail_channel_id or str(channel.id) == jail_channel_id:
        return
    role = discord.utils.get(guild.roles, name=JAIL_ROLE_NAME)
    if not role:
        return
    try:
        await channel.set_permissions(role, view_channel=False, send_messages=False, connect=False, reason="Jail system -- auto-lock new channel")
    except Exception as e:
        print(f"⚠️ Failed to auto-lock new channel {channel.id} for jail role: {e}")

# --------------------------------------------------------
# 🗣️ NATURAL-LANGUAGE CONTROL CHANNEL
# --------------------------------------------------------
# Lets Administrators type plain-English instructions in one designated
# channel (?setcontrolchannel) instead of exact bot commands -- e.g.
# "timeout @user for 1 hour", "disable spam protection for 1hr in #general",
# "unjail @user after 10 mins". An AI call classifies the instruction into a
# small fixed set of supported actions plus any duration/delay/reason/amount
# mentioned; @user and #channel targets are taken directly from Discord's
# own resolved message.mentions/channel_mentions (never from the AI) so a
# hallucinated ID can never end up targeting the wrong person.
#
# "duration" = how long the EFFECT should last (e.g. timeout length, or how
# long a toggle stays off before automatically reverting -- handled via
# scheduling the opposite action, see perform_control_action below).
# "delay" = how long to WAIT before performing the action at all (e.g.
# "after 10 minutes") -- handled by scheduled_control_actions + the loop
# below, so it survives a restart just like reminders/timecapsules do.
CONTROL_ACTIONS = {
    "timeout", "kick", "ban", "warn", "jail", "unjail",
    "disable_roast", "enable_roast",
    "disable_spam_protection", "enable_spam_protection",
    "lock_channel", "unlock_channel", "purge",
}
CONTROL_ACTIONS_NEEDING_MEMBER = {"timeout", "kick", "ban", "warn", "jail", "unjail"}
CONTROL_ACTIONS_NEEDING_CHANNEL = {"lock_channel", "unlock_channel", "purge"}

def save_scheduled_control_actions():
    save_data(scheduled_control_actions, SCHEDULED_CONTROL_ACTIONS_FILE)

def strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def build_control_prompt(instruction: str) -> str:
    return (
        "You are a command parser for a Discord moderation bot's natural-language control channel. "
        "A server administrator will describe what they want done in plain English. Reply with STRICT "
        "JSON only, matching exactly this schema, nothing else, no markdown code fences:\n"
        '{"action": "<one of: timeout, kick, ban, warn, jail, unjail, disable_roast, enable_roast, '
        'disable_spam_protection, enable_spam_protection, lock_channel, unlock_channel, purge, unsupported>", '
        '"duration_minutes": <number or null -- how long an EFFECT should last, e.g. a timeout length, or '
        'how long a toggle should stay off before automatically reverting>, '
        '"delay_minutes": <number or null -- how long to WAIT before performing the action at all, e.g. '
        '"after 10 minutes" means delay_minutes: 10>, "reason": "<short string or null>", '
        '"amount": <number or null, only used for purge>}\n\n'
        "Convert any time units mentioned (hours, days) into minutes for both duration_minutes and "
        "delay_minutes. Ignore any @mentions or #channel-mentions in the message -- those are already "
        "resolved separately, just focus on classifying the action and extracting numbers/duration/reason. "
        "If the instruction doesn't clearly match one of the actions above, use \"unsupported\".\n\n"
        f"Instruction: \"{instruction}\""
    )

async def control_action_timeout(target_member: discord.Member, duration_minutes, reason: str) -> str:
    minutes = duration_minutes or 10
    try:
        await target_member.timeout(discord.utils.utcnow() + timedelta(minutes=minutes), reason=reason)
        return f"✅ Timed out {target_member.mention} for {minutes} minute(s). Reason: {reason}"
    except discord.Forbidden:
        return "❌ I don't have permission to timeout that member."
    except Exception as e:
        return f"❌ Failed to timeout: {e}"

async def control_action_kick(target_member: discord.Member, reason: str) -> str:
    try:
        await target_member.kick(reason=reason)
        return f"👢 Kicked {target_member.mention}. Reason: {reason}"
    except discord.Forbidden:
        return "❌ I don't have permission to kick that member."

async def control_action_ban(guild: discord.Guild, target_member: discord.Member, reason: str) -> str:
    try:
        await guild.ban(target_member, reason=reason)
        return f"🔨 Banned {target_member.mention}. Reason: {reason}"
    except discord.Forbidden:
        return "❌ I don't have permission to ban that member."

async def control_action_warn(guild: discord.Guild, target_member: discord.Member, reason: str, moderator: discord.abc.User) -> str:
    add_warning(guild.id, str(target_member.id), reason, moderator.id, moderator.name)
    try:
        await target_member.send(f"⚠️ You were warned in **{guild.name}**. Reason: {reason}")
    except Exception:
        pass
    return f"⚠️ Warned {target_member.mention}. Reason: {reason}"

async def control_action_jail(guild: discord.Guild, target_member: discord.Member, reason: str) -> str:
    jail_channel_id = server_config.get(str(guild.id), {}).get("jail_channel_id")
    if not jail_channel_id:
        return "❌ No jail channel configured -- run `?setjailchannel` first."
    role = discord.utils.get(guild.roles, name=JAIL_ROLE_NAME)
    if not role:
        return "❌ The Jailed role doesn't exist yet -- run `?setjailchannel` first."
    if role in target_member.roles:
        return f"❌ {target_member.display_name} is already jailed."
    try:
        await target_member.add_roles(role, reason=reason)
    except discord.Forbidden:
        return "❌ I can't assign the Jailed role -- check my role position."
    if target_member.voice and target_member.voice.channel:
        try:
            await target_member.move_to(None, reason="Jailed via control channel")
        except Exception:
            pass
    return f"🔒 Jailed {target_member.mention}. Reason: {reason}"

async def control_action_unjail(guild: discord.Guild, target_member: discord.Member) -> str:
    role = discord.utils.get(guild.roles, name=JAIL_ROLE_NAME)
    if not role or role not in target_member.roles:
        return f"❌ {target_member.display_name} isn't currently jailed."
    try:
        await target_member.remove_roles(role, reason="Released via control channel")
    except discord.Forbidden:
        return "❌ I don't have permission to remove that role."
    return f"🔓 Released {target_member.mention} from jail."

async def control_action_disable_roast(guild: discord.Guild) -> str:
    gid = str(guild.id)
    server_config.setdefault(gid, {})["random_roast_disabled"] = True
    save_data(server_config, SERVER_CONFIG_FILE)
    return "🔇 Random roast disabled."

async def control_action_enable_roast(guild: discord.Guild) -> str:
    gid = str(guild.id)
    server_config.setdefault(gid, {})["random_roast_disabled"] = False
    save_data(server_config, SERVER_CONFIG_FILE)
    return "🔊 Random roast re-enabled."

async def control_action_disable_spam(guild: discord.Guild, target_channel: Optional[discord.TextChannel], duration_minutes) -> str:
    gid = str(guild.id)
    server_config.setdefault(gid, {})
    value = (datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)).isoformat() if duration_minutes else True
    if target_channel:
        server_config[gid].setdefault("spam_protection_disabled_channels", {})
        server_config[gid]["spam_protection_disabled_channels"][str(target_channel.id)] = value
        scope = f"in {target_channel.mention}"
    else:
        server_config[gid]["spam_protection_disabled_until"] = value
        scope = "server-wide"
    save_data(server_config, SERVER_CONFIG_FILE)
    return f"🔇 Spam protection disabled {scope}."

async def control_action_enable_spam(guild: discord.Guild, target_channel: Optional[discord.TextChannel]) -> str:
    gid = str(guild.id)
    server_config.setdefault(gid, {})
    if target_channel:
        server_config[gid].setdefault("spam_protection_disabled_channels", {}).pop(str(target_channel.id), None)
        scope = f"in {target_channel.mention}"
    else:
        server_config[gid].pop("spam_protection_disabled_until", None)
        scope = "server-wide"
    save_data(server_config, SERVER_CONFIG_FILE)
    return f"🔊 Spam protection re-enabled {scope}."

async def control_action_lock_channel(target_channel: discord.TextChannel, reason: str) -> str:
    try:
        await target_channel.set_permissions(target_channel.guild.default_role, send_messages=False, reason=reason)
        return f"🔒 Locked {target_channel.mention}."
    except discord.Forbidden:
        return "❌ I don't have permission to edit that channel's permissions."

async def control_action_unlock_channel(target_channel: discord.TextChannel) -> str:
    try:
        await target_channel.set_permissions(target_channel.guild.default_role, send_messages=True)
        return f"🔓 Unlocked {target_channel.mention}."
    except discord.Forbidden:
        return "❌ I don't have permission to edit that channel's permissions."

async def control_action_purge(target_channel: discord.TextChannel, amount, guild_id: Optional[int] = None, moderator_id: Optional[int] = None) -> str:
    amount = min(max(int(amount or 10), 1), 100)
    try:
        deleted = await target_channel.purge(limit=amount)
        if guild_id and moderator_id:
            record_undo(guild_id, "purge", moderator_id, {
                "channel_id": target_channel.id,
                "messages": [
                    {"author_name": m.author.display_name, "content": m.content, "is_bot": m.author.bot}
                    for m in reversed(deleted)
                ],
            })
        return f"🗑️ Deleted {len(deleted)} messages in {target_channel.mention}."
    except discord.Forbidden:
        return "❌ I don't have permission to delete messages there."

async def perform_control_action(
    guild: discord.Guild, action: str, *,
    target_member: Optional[discord.Member] = None,
    target_channel: Optional[discord.TextChannel] = None,
    duration_minutes=None, reason: str = "No reason provided", amount=None,
    origin_channel_id: Optional[int] = None, moderator_id: Optional[int] = None,
) -> str:
    if action == "timeout":
        result = await control_action_timeout(target_member, duration_minutes, reason)
        if result.startswith("✅") and moderator_id:
            record_undo(guild.id, "timeout", moderator_id, {"user_id": target_member.id})
        return result
    if action == "kick":
        role_ids = [r.id for r in target_member.roles if r.name != "@everyone"] if target_member else []
        result = await control_action_kick(target_member, reason)
        if result.startswith("👢") and moderator_id:
            record_undo(guild.id, "kick", moderator_id, {"user_id": target_member.id, "role_ids": role_ids})
        return result
    if action == "ban":
        result = await control_action_ban(guild, target_member, reason)
        if result.startswith("🔨") and moderator_id:
            record_undo(guild.id, "ban", moderator_id, {"user_id": target_member.id})
        return result
    if action == "warn":
        moderator = (guild.get_member(moderator_id) if moderator_id else None) or guild.me
        result = await control_action_warn(guild, target_member, reason, moderator)
        if result.startswith("⚠️") and moderator_id:
            record_undo(guild.id, "warn", moderator_id, {"user_id": target_member.id})
        return result
    if action == "jail":
        result = await control_action_jail(guild, target_member, reason)
        if result.startswith("🔒"):
            if moderator_id:
                record_undo(guild.id, "jail", moderator_id, {"user_id": target_member.id})
            if duration_minutes:
                await schedule_control_action(
                    guild.id, origin_channel_id, "unjail",
                    target_member.id if target_member else None, None,
                    None, "Auto-release after scheduled duration", None, duration_minutes, moderator_id
                )
                result += f" Will auto-release in {duration_minutes} minute(s)."
        return result
    if action == "unjail":
        return await control_action_unjail(guild, target_member)
    if action == "disable_roast":
        result = await control_action_disable_roast(guild)
        if duration_minutes:
            await schedule_control_action(guild.id, origin_channel_id, "enable_roast", None, None, None, "Auto re-enable", None, duration_minutes, moderator_id)
            result += f" Re-enabling automatically in {duration_minutes} minute(s)."
        return result
    if action == "enable_roast":
        return await control_action_enable_roast(guild)
    if action == "disable_spam_protection":
        result = await control_action_disable_spam(guild, target_channel, duration_minutes)
        if duration_minutes:
            result += f" Re-enabling automatically in {duration_minutes} minute(s)."
        return result
    if action == "enable_spam_protection":
        return await control_action_enable_spam(guild, target_channel)
    if action == "lock_channel":
        result = await control_action_lock_channel(target_channel, reason)
        if result.startswith("🔒"):
            if moderator_id:
                record_undo(guild.id, "lock_channel", moderator_id, {"channel_id": target_channel.id})
            if duration_minutes:
                await schedule_control_action(guild.id, origin_channel_id, "unlock_channel", None, target_channel.id, None, "Auto-unlock", None, duration_minutes, moderator_id)
                result += f" Will auto-unlock in {duration_minutes} minute(s)."
        return result
    if action == "unlock_channel":
        result = await control_action_unlock_channel(target_channel)
        if result.startswith("🔓") and moderator_id:
            record_undo(guild.id, "unlock_channel", moderator_id, {"channel_id": target_channel.id})
        return result
    if action == "purge":
        return await control_action_purge(target_channel, amount, guild_id=guild.id, moderator_id=moderator_id)
    return "❌ I don't know how to do that yet."

async def schedule_control_action(guild_id, origin_channel_id, action, target_user_id, target_channel_id, duration_minutes, reason, amount, delay_minutes, moderator_id):
    action_id = str(datetime.now().timestamp())
    run_at = (datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)).isoformat()
    scheduled_control_actions[action_id] = {
        "guild_id": guild_id, "origin_channel_id": origin_channel_id, "action": action,
        "target_user_id": target_user_id, "target_channel_id": target_channel_id,
        "duration_minutes": duration_minutes, "reason": reason, "amount": amount,
        "run_at": run_at, "moderator_id": moderator_id,
    }
    save_scheduled_control_actions()

async def execute_scheduled_control_action(data: dict):
    guild = bot.get_guild(data["guild_id"])
    if not guild:
        return
    origin_channel = guild.get_channel(data["origin_channel_id"]) if data.get("origin_channel_id") else None
    target_member = guild.get_member(data["target_user_id"]) if data.get("target_user_id") else None
    target_channel = guild.get_channel(data["target_channel_id"]) if data.get("target_channel_id") else None

    result = await perform_control_action(
        guild, data["action"], target_member=target_member, target_channel=target_channel,
        duration_minutes=data.get("duration_minutes"), reason=data.get("reason") or "No reason provided",
        amount=data.get("amount"), origin_channel_id=data.get("origin_channel_id"), moderator_id=data.get("moderator_id"),
    )
    if origin_channel:
        try:
            await origin_channel.send(f"⏰ Scheduled action executed: {result}")
        except Exception:
            pass

@tasks.loop(minutes=1.0)
async def scheduled_control_action_loop():
    now = datetime.now(timezone.utc)
    due_ids = []
    for action_id, data in list(scheduled_control_actions.items()):
        try:
            run_at = datetime.fromisoformat(data["run_at"])
        except Exception:
            due_ids.append(action_id)
            continue
        if run_at <= now:
            bot.loop.create_task(execute_scheduled_control_action(data))
            due_ids.append(action_id)
    for aid in due_ids:
        scheduled_control_actions.pop(aid, None)
    if due_ids:
        save_scheduled_control_actions()

@scheduled_control_action_loop.error
async def scheduled_control_action_loop_error(error):
    print(f"🚨 Scheduled Control Action Loop Error: {error}")

async def handle_control_instruction(message: discord.Message):
    guild = message.guild
    target_member = next((m for m in message.mentions if not m.bot), None)
    target_channel = message.channel_mentions[0] if message.channel_mentions else None

    async with message.channel.typing():
        try:
            raw = await asyncio.to_thread(get_groq_text, build_control_prompt(message.content))
            data = json.loads(strip_json_fence(raw))
        except Exception as e:
            print(f"⚠️ Control channel parse failed: {e}")
            return await message.reply("❌ I couldn't understand that instruction, try rephrasing it.")

    action = data.get("action")
    if action not in CONTROL_ACTIONS:
        return await message.reply("❌ I don't know how to do that yet -- try rephrasing, or use the specific `?command` for it.")

    duration_minutes = data.get("duration_minutes")
    delay_minutes = data.get("delay_minutes")
    reason = data.get("reason") or f"Requested via control channel by {message.author}"
    amount = data.get("amount")

    if action in CONTROL_ACTIONS_NEEDING_MEMBER and not target_member:
        return await message.reply("❌ Mention who this should apply to (e.g. `timeout @user for 1 hour`).")

    effective_channel = target_channel or message.channel

    if delay_minutes and delay_minutes > 0:
        await schedule_control_action(
            guild.id, message.channel.id, action,
            target_member.id if target_member else None,
            effective_channel.id if (action in CONTROL_ACTIONS_NEEDING_CHANNEL or "spam_protection" in action) else None,
            duration_minutes, reason, amount, delay_minutes, message.author.id
        )
        return await message.reply(f"⏳ Got it -- I'll do that in **{delay_minutes} minute(s)**.")

    result = await perform_control_action(
        guild, action, target_member=target_member, target_channel=effective_channel,
        duration_minutes=duration_minutes, reason=reason, amount=amount,
        origin_channel_id=message.channel.id, moderator_id=message.author.id,
    )
    await message.reply(result)

@bot.command(name="setcontrolchannel", usage="#channel", help="[Admins only] Sets a channel where plain-English moderation instructions (e.g. 'timeout @user for 1 hour') get carried out automatically.")
@commands.has_permissions(administrator=True)
async def set_control_channel(ctx, channel: discord.TextChannel):
    gid = str(ctx.guild.id)
    server_config.setdefault(gid, {})["control_channel_id"] = str(channel.id)
    save_data(server_config, SERVER_CONFIG_FILE)
    await ctx.send(
        f"✅ {channel.mention} is now the control channel. Administrators can type plain-English "
        f"instructions there, e.g.:\n"
        f"• `timeout @user for 1 hour`\n"
        f"• `disable spam protection for 1hr in #general`\n"
        f"• `unjail @user after 10 mins`\n"
        f"• `lock this channel for 30 minutes`\n"
        f"• `warn @user for spamming`\n"
        f"Supported right now: timeout, kick, ban, warn, jail, unjail, enable/disable random roast, "
        f"enable/disable spam protection, lock/unlock a channel, purge messages. Anything else gets a "
        f"'don't know how to do that yet' reply instead of guessing."
    )
    await send_mod_log(ctx.guild, "Control Channel Set", f"**Channel:** {channel.mention}", ctx.author)

# --------------------------------------------------------
# ↩️ UNDO SYSTEM (?undo)
# --------------------------------------------------------
# Tracks the single most recent undoable action PER SERVER (not per-user) --
# whoever has the right permission can undo whatever just happened, same as
# most mod bots' undo/rollback commands. Covers actions from both the
# classic ?commands above AND the natural-language control channel, since
# perform_control_action() calls record_undo() too. In-memory only (not
# persisted) -- a restart clears the undo slot, which is an acceptable
# tradeoff since undoing something from before a redeploy is rarely useful.
last_undoable_action = {}  # { guild_id: {"type": str, "performed_by": int, "timestamp": iso, "data": {...}} }
# { (guild_id, user_id): [role_id, ...] } -- roles to restore if a kicked
# member rejoins after their kick was undone (see undo_kick + on_member_join).
pending_kick_restores = {}

def record_undo(guild_id: int, action_type: str, performed_by: int, data: dict):
    last_undoable_action[guild_id] = {
        "type": action_type,
        "performed_by": performed_by,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

async def undo_purge(guild: discord.Guild, data: dict) -> str:
    channel = guild.get_channel(data["channel_id"])
    if not channel:
        return "❌ Can't undo -- that channel no longer exists."
    messages = data.get("messages", [])
    if not messages:
        return "❌ Nothing to restore -- no message content was captured."
    lines = [f"**{m['author_name']}{' 🤖' if m['is_bot'] else ''}:** {m['content'] or '*(no text content)*'}" for m in messages]
    await safe_send(
        channel,
        f"↩️ **Restoring {len(messages)} purged message(s)** -- Discord has no true 'undelete', "
        f"so these are recreated copies, not the original messages:\n\n" + "\n".join(lines)
    )
    return f"↩️ Reposted {len(messages)} purged message(s) in {channel.mention}."

async def undo_warn(guild: discord.Guild, data: dict) -> str:
    gid, uid = str(guild.id), str(data["user_id"])
    warns = warnings_data.get(gid, {}).get(uid)
    if not warns:
        return "❌ Nothing to undo -- no warnings found for that user."
    removed = warns.pop()
    save_data(warnings_data, WARNINGS_FILE)
    member = guild.get_member(int(uid))
    name = member.display_name if member else f"user {uid}"
    return f"↩️ Removed the most recent warning from {name} (\"{removed.get('reason', '')}\")."

async def undo_timeout(guild: discord.Guild, data: dict) -> str:
    member = guild.get_member(data["user_id"])
    if not member:
        return "❌ That member is no longer in the server."
    try:
        await member.timeout(None, reason="Undo -- timeout removed")
        return f"↩️ Removed timeout from {member.mention}."
    except discord.Forbidden:
        return "❌ I don't have permission to remove that timeout."

async def undo_ban(guild: discord.Guild, data: dict) -> str:
    try:
        await guild.unban(discord.Object(id=data["user_id"]), reason="Undo -- ban reversed")
        return f"↩️ Unbanned <@{data['user_id']}>."
    except discord.NotFound:
        return "❌ That user isn't currently banned."
    except discord.Forbidden:
        return "❌ I don't have permission to unban."

async def undo_kick(guild: discord.Guild, data: dict) -> str:
    # Kicks can't be truly undone (Discord has no "un-kick" -- they'd need a
    # fresh invite to rejoin), so this is best-effort: generate a one-time
    # invite, try to DM it to them, and queue their old roles to be restored
    # automatically if/when they use it to rejoin.
    user_id = data["user_id"]
    pending_kick_restores[(guild.id, user_id)] = data.get("role_ids", [])

    invite = None
    for channel in guild.text_channels:
        try:
            invite = await channel.create_invite(max_uses=1, unique=True, reason="Undo kick -- inviting them back")
            break
        except Exception:
            continue

    dm_sent = False
    if invite:
        try:
            user_obj = await bot.fetch_user(user_id)
            await user_obj.send(f"You were kicked from **{guild.name}**, but a moderator undid it. Rejoin here: {invite.url}")
            dm_sent = True
        except Exception:
            pass

    if not invite:
        return "❌ Kicks can't be truly undone, and I couldn't create an invite link -- you'll need to invite them back manually. Their roles will restore automatically if they rejoin."
    note = " (DM sent to them)" if dm_sent else " (couldn't DM them -- share this yourself)"
    return f"↩️ Kicks can't be truly undone, but here's a one-time invite{note}: {invite.url}\nTheir roles will be restored automatically if they rejoin."

async def undo_jail(guild: discord.Guild, data: dict) -> str:
    member = guild.get_member(data["user_id"])
    if not member:
        return "❌ That member is no longer in the server."
    return await control_action_unjail(guild, member)

async def undo_lock_channel(guild: discord.Guild, data: dict) -> str:
    channel = guild.get_channel(data["channel_id"])
    if not channel:
        return "❌ That channel no longer exists."
    return await control_action_unlock_channel(channel)

async def undo_unlock_channel(guild: discord.Guild, data: dict) -> str:
    channel = guild.get_channel(data["channel_id"])
    if not channel:
        return "❌ That channel no longer exists."
    return await control_action_lock_channel(channel, "Undo -- re-locked")

async def undo_resetlb(guild: discord.Guild, data: dict) -> str:
    gid, uid = str(guild.id), str(data["user_id"])
    message_counts.setdefault(gid, {})[uid] = data["old_count"]
    save_data(message_counts, MESSAGES_FILE)
    member = guild.get_member(int(uid))
    name = member.display_name if member else f"user {uid}"
    return f"↩️ Restored {name}'s weekly leaderboard count to **{data['old_count']}**."

UNDO_HANDLERS = {
    "purge": undo_purge,
    "warn": undo_warn,
    "timeout": undo_timeout,
    "ban": undo_ban,
    "kick": undo_kick,
    "jail": undo_jail,
    "lock_channel": undo_lock_channel,
    "unlock_channel": undo_unlock_channel,
    "resetlb": undo_resetlb,
}

@bot.command(name="undo", help="[Mods only] Reverses the last undoable moderation action on this server (purge, warn, timeout, ban, kick, jail, lock/unlock, resetlb).")
@commands.has_permissions(manage_messages=True)
async def undo_last_action(ctx):
    record = last_undoable_action.get(ctx.guild.id)
    if not record:
        return await ctx.send("❌ Nothing to undo.")

    handler = UNDO_HANDLERS.get(record["type"])
    if not handler:
        return await ctx.send(f"❌ The last action (`{record['type']}`) can't be undone.")

    result = await handler(ctx.guild, record["data"])
    del last_undoable_action[ctx.guild.id]
    await ctx.send(result)
    await send_mod_log(
        ctx.guild, "Action Undone",
        f"**Type:** {record['type']}\n**Originally by:** <@{record['performed_by']}>\n**Undone by:** {ctx.author.mention}\n**Result:** {result}",
        ctx.author
    )

# --- WARNING SYSTEM COMMANDS ---

@bot.command(name="warnings", help="View a member's warnings on this server.")
@commands.has_permissions(kick_members=True)
async def view_warnings(ctx, member: discord.Member):
    gid = str(ctx.guild.id)
    user_id = str(member.id)
    guild_warnings = warnings_data.get(gid, {})

    if user_id not in guild_warnings or not isinstance(guild_warnings[user_id], list) or len(guild_warnings[user_id]) == 0:
        return await ctx.send(f"✅ **{member.display_name}** has a clean record on this server.")

    embed = discord.Embed(
        title=f"⚠️ Warning History: {member.display_name}",
        color=discord.Color.orange()
    )
    embed.set_thumbnail(url=member.display_avatar.url)

    for i, warn in enumerate(guild_warnings[user_id], 1):
        reason = warn.get('reason', 'No reason provided')
        mod_name = warn.get('moderator_name', 'Unknown Admin')
        date = warn.get('date', 'Unknown Date')
        
        embed.add_field(
            name=f"Warning #{i}",
            value=f"**Reason:** {reason}\n**By:** {mod_name}\n**Date:** {date}",
            inline=False
        )

    await ctx.send(embed=embed)

@bot.command(name="delwarn", help="Delete a specific warning by ID or 'all'. Usage: +delwarn @user 1")
@commands.has_permissions(manage_messages=True)
async def delete_warning(ctx, member: discord.Member, warn_id: str):
    gid = str(ctx.guild.id)
    user_id = str(member.id)
    guild_warnings = warnings_data.setdefault(gid, {})

    if user_id not in guild_warnings or not guild_warnings[user_id]:
        return await ctx.send("This user has no warnings to delete on this server.")

    if warn_id.lower() == "all":
        guild_warnings[user_id] = []
        save_data(warnings_data, WARNINGS_FILE)
        await ctx.send(f"🗑️ Cleared all warnings for **{member.display_name}** on this server.")
        await send_mod_log(ctx.guild, "All Warnings Cleared", f"**User:** {member.mention} ({member.id})", ctx.author)
        return

    try:
        idx = int(warn_id) - 1
        if 0 <= idx < len(guild_warnings[user_id]):
            removed = guild_warnings[user_id].pop(idx)
            save_data(warnings_data, WARNINGS_FILE)
            await ctx.send(f"✅ Deleted Warning #{warn_id} ({removed['reason']}) for {member.mention}.")
            await send_mod_log(ctx.guild, "Warning Deleted", f"**User:** {member.mention} ({member.id})\n**Removed Reason:** {removed['reason']}", ctx.author)
        else:
            await ctx.send(f"❌ Invalid ID. Use `?warnings @user` to see valid IDs.")
    except ValueError:
        await ctx.send("❌ Please provide a valid number ID or type `all`.")        

#-------BIRTHDAY WISH------
@bot.command(name='wish', help="Wish someone a birthday and give them a temporary role. (Mods only)")
@commands.has_permissions(manage_roles=True)
async def wish(ctx, member: discord.Member):
    MALE_ROLE = "Male"
    FEMALE_ROLE = "Female"
    BDAY_BOY = "Birthday Boy"
    BDAY_GIRL = "Birthday Girl"

    role_to_give_name = BDAY_BOY if any(r.name == MALE_ROLE for r in member.roles) else BDAY_GIRL
    
    bday_role = discord.utils.get(ctx.guild.roles, name=role_to_give_name)
    if not bday_role:
        bday_role = await ctx.guild.create_role(name=role_to_give_name, color=discord.Color.magenta())

    await member.add_roles(bday_role)

    embed = discord.Embed(
        title="🎉 Happy Birthday! 🎉",
        description=f"Everyone join us in wishing {member.mention} a very Happy Birthday! 🎂✨\n\nYou've been granted the **{role_to_give_name}** role for 24 hours!",
        color=discord.Color.random()
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    
    await ctx.send(content=f"🎈 {member.mention} 🎈", embed=embed)
    await send_mod_log(ctx.guild, "Birthday Wish Given", f"**User:** {member.mention}\n**Role:** {role_to_give_name}", ctx.author)

    async def remove_bday_role():
        await asyncio.sleep(86400)
        if bday_role in member.roles:
            await member.remove_roles(bday_role)
            print(f"Removed birthday role from {member.name}")

    bot.loop.create_task(remove_bday_role())

# --------------------------------------------------------
# 💍 MARRIAGE & FRIENDSHIP SYSTEM
# --------------------------------------------------------

class ConfirmRelationshipView(ui.View):
    """Generic Accept/Decline view used for both marriage and friend requests.
    Only the targeted user can respond, and it expires after 2 minutes."""
    def __init__(self, requester: discord.Member, target: discord.Member, kind: str):
        super().__init__(timeout=120)
        self.requester = requester
        self.target = target
        self.kind = kind
        self.responded = False
        self.message = None

    async def on_timeout(self):
        if self.responded:
            return
        for child in self.children:
            child.disabled = True
        if self.message:
            try:
                word = "marriage" if self.kind == "marry" else "friend"
                await self.message.edit(content=f"⌛ This {word} request expired.", view=self)
            except Exception:
                pass

    @ui.button(label="Accept", style=discord.ButtonStyle.success, emoji="✅")
    async def accept(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user.id != self.target.id:
            return await interaction.response.send_message("This request isn't for you!", ephemeral=True)
        self.responded = True
        self.disable_all()

        if self.kind == "marry":
            guild_marriages = get_guild_marriages(interaction.guild.id)
            guild_marriages[str(self.requester.id)] = str(self.target.id)
            guild_marriages[str(self.target.id)] = str(self.requester.id)
            save_data(marriages, MARRIAGES_FILE)

            guild_marriage_stats = get_guild_marriage_stats(interaction.guild.id)
            guild_marriage_stats[str(self.requester.id)] = guild_marriage_stats.get(str(self.requester.id), 0) + 1
            guild_marriage_stats[str(self.target.id)] = guild_marriage_stats.get(str(self.target.id), 0) + 1
            save_data(marriage_stats, MARRIAGE_STATS_FILE)

            await interaction.response.edit_message(
                content=f"💍 **{self.requester.mention} and {self.target.mention} are now married!** 🎉",
                view=self
            )
        else:
            uid, tid = str(self.requester.id), str(self.target.id)
            guild_friends = get_guild_friends(interaction.guild.id)
            guild_friends.setdefault(uid, [])
            guild_friends.setdefault(tid, [])
            if tid not in guild_friends[uid]:
                guild_friends[uid].append(tid)
            if uid not in guild_friends[tid]:
                guild_friends[tid].append(uid)
            save_data(friends, FRIENDS_FILE)

            await interaction.response.edit_message(
                content=f"🤝 **{self.requester.mention} and {self.target.mention} are now friends!**",
                view=self
            )

    @ui.button(label="Decline", style=discord.ButtonStyle.danger, emoji="❌")
    async def decline(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user.id != self.target.id:
            return await interaction.response.send_message("This request isn't for you!", ephemeral=True)
        self.responded = True
        self.disable_all()
        word = "marriage" if self.kind == "marry" else "friend"
        await interaction.response.edit_message(content=f"💔 {self.target.mention} declined the {word} request.", view=self)

    def disable_all(self):
        for child in self.children:
            child.disabled = True
        self.stop()

# --------------------------------------------------------
# 💘 SHIP & COMPATIBILITY
# --------------------------------------------------------
SHIP_MARRY_THRESHOLD = 85

def compute_compatibility(user_a_id: int, user_b_id: int) -> int:
    pair_key = "-".join(sorted([str(user_a_id), str(user_b_id)]))
    digest = hashlib.sha256(pair_key.encode()).hexdigest()
    return int(digest, 16) % 101

def build_compatibility_prompt(user_a: discord.Member, user_b: discord.Member, score: int) -> str:
    return (
        f"Two Discord users, {user_a.display_name} and {user_b.display_name}, have a compatibility "
        f"score of {score}%. Write one short, funny, playful sentence explaining why (or why not) "
        f"they're compatible. Output only the sentence, nothing else."
    )

async def generate_compatibility_blurb(user_a: discord.Member, user_b: discord.Member, score: int) -> str:
    prompt = build_compatibility_prompt(user_a, user_b, score)
    try:
        return await asyncio.to_thread(get_groq_text, prompt)
    except Exception as e:
        print(f"⚠️ Compatibility blurb generation failed: {e}")
        return "The stars are unclear on this one... 🌌"

class DivorceConsentView(ui.View):
    def __init__(self, person_wanting_out: discord.Member, current_partner: discord.Member):
        super().__init__(timeout=120)
        self.person_wanting_out = person_wanting_out
        self.current_partner = current_partner
        self.result = None
        self.message = None

    @ui.button(label="Agree to Divorce", style=discord.ButtonStyle.danger, emoji="💔")
    async def agree(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user.id != self.current_partner.id:
            return await interaction.response.send_message("This request isn't for you!", ephemeral=True)
        self.result = True
        self._disable()
        await interaction.response.edit_message(content=f"💔 {self.current_partner.mention} agreed to the divorce.", view=self)
        self.stop()

    @ui.button(label="Refuse", style=discord.ButtonStyle.secondary)
    async def refuse(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user.id != self.current_partner.id:
            return await interaction.response.send_message("This request isn't for you!", ephemeral=True)
        self.result = False
        self._disable()
        await interaction.response.edit_message(content=f"❌ {self.current_partner.mention} refused to divorce.", view=self)
        self.stop()

    def _disable(self):
        for child in self.children:
            child.disabled = True

    async def on_timeout(self):
        if self.result is not None:
            return
        self.result = False
        self._disable()
        if self.message:
            try:
                await self.message.edit(content="⌛ Divorce request expired (no response).", view=self)
            except Exception:
                pass

async def resolve_existing_marriage(channel, person: discord.Member) -> bool:
    guild = channel.guild
    guild_marriages = get_guild_marriages(guild.id)
    uid = str(person.id)
    if uid not in guild_marriages:
        return True

    partner_id = int(guild_marriages[uid])
    partner = guild.get_member(partner_id)
    if not partner:
        guild_marriages.pop(uid, None)
        guild_marriages.pop(str(partner_id), None)
        save_data(marriages, MARRIAGES_FILE)
        return True

    consent_view = DivorceConsentView(person, partner)
    consent_view.message = await channel.send(
        f"💔 {partner.mention}, {person.mention} wants to marry someone else and needs to divorce you first. Do you agree?",
        view=consent_view
    )
    await consent_view.wait()

    if not consent_view.result:
        return False

    guild_marriages.pop(uid, None)
    guild_marriages.pop(str(partner_id), None)
    save_data(marriages, MARRIAGES_FILE)
    return True

class ShipMarryView(ui.View):
    def __init__(self, user_a: discord.Member, user_b: discord.Member, base_prompt: str, embed_builder, blurb: str):
        super().__init__(timeout=180)
        self.user_a = user_a
        self.user_b = user_b
        self.base_prompt = base_prompt
        self.embed_builder = embed_builder
        self.current_lang = "en"
        self.message = None
        self._set_lang_button_label()

    def _set_lang_button_label(self):
        self.language_toggle.label = "🇬🇧 Switch to English" if self.current_lang == "hi" else "🇮🇳 Switch to Hinglish"

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass

    @ui.button(label="💍 Marry?", style=discord.ButtonStyle.success)
    async def marry_button(self, interaction: discord.Interaction, button: ui.Button):
        clicker = interaction.user
        if clicker.id not in (self.user_a.id, self.user_b.id):
            return await interaction.response.send_message("Only the two people being shipped can use this button!", ephemeral=True)

        other = self.user_b if clicker.id == self.user_a.id else self.user_a

        if get_guild_marriages(interaction.guild.id).get(str(clicker.id)) == str(other.id):
            return await interaction.response.send_message("You're already married to them! 💍", ephemeral=True)

        await interaction.response.defer()
        channel = interaction.channel

        if not await resolve_existing_marriage(channel, clicker):
            return await channel.send(f"❌ Marriage attempt cancelled -- {clicker.display_name}'s existing partner didn't agree to the divorce.")

        if not await resolve_existing_marriage(channel, other):
            return await channel.send(f"❌ Marriage attempt cancelled -- {other.display_name}'s existing partner didn't agree to the divorce.")

        marry_view = ConfirmRelationshipView(clicker, other, "marry")
        marry_view.message = await channel.send(
            f"💍 {other.mention}, {clicker.mention} wants to marry you! Do you accept?",
            view=marry_view
        )

    @ui.button(label="🇮🇳 Switch to Hinglish", style=discord.ButtonStyle.secondary)
    async def language_toggle(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.defer()
        target_lang = "English" if self.current_lang == "hi" else "Hinglish"
        lang_instruction = (
            "\n\nWrite this entirely in natural, casual Hinglish (the way people actually type it in "
            "Indian Discord/WhatsApp chats -- Hindi and English mixed together, informal, not formal "
            "textbook Hindi)." if target_lang == "Hinglish"
            else "\n\nWrite this entirely in natural English."
        )
        try:
            new_text = await asyncio.to_thread(get_groq_text, self.base_prompt + lang_instruction)
        except Exception as e:
            print(f"⚠️ Ship language toggle failed: {e}")
            return await interaction.followup.send("❌ Couldn't switch language right now.", ephemeral=True)

        self.current_lang = "hi" if target_lang == "Hinglish" else "en"
        self._set_lang_button_label()
        new_embed = self.embed_builder(new_text.strip())
        try:
            await interaction.message.edit(embed=new_embed, view=self)
        except Exception as e:
            print(f"⚠️ Failed to edit ship message after language toggle: {e}")

@bot.command(usage="[user1] [user2]", help="Ships two users with a compatibility %. 85%+ unlocks a Marry button.")
async def ship(ctx, user1: discord.Member = None, user2: discord.Member = None):
    if user1 is None:
        return await ctx.send("❌ Mention at least one person. Usage: `?ship @user` or `?ship @user1 @user2`")
    if user2 is None:
        user_a, user_b = ctx.author, user1
    else:
        user_a, user_b = user1, user2

    if user_a.id == user_b.id:
        return await ctx.send("❌ Can't ship someone with themselves!")

    score = compute_compatibility(user_a.id, user_b.id)
    prompt = build_compatibility_prompt(user_a, user_b, score)
    blurb = await generate_compatibility_blurb(user_a, user_b, score)

    bar_filled = "💗" * (score // 10)
    bar_empty = "🤍" * (10 - score // 10)

    def build_embed(text):
        embed = discord.Embed(
            title=f"💘 {user_a.display_name} + {user_b.display_name}",
            description=f"{bar_filled}{bar_empty}\n**{score}% Compatible**\n\n*{text}*",
            color=discord.Color.from_rgb(255, 105, 180)
        )
        if score >= SHIP_MARRY_THRESHOLD:
            embed.set_footer(text="85%+ compatible! Either of you can hit the button below.")
        return embed

    if score >= SHIP_MARRY_THRESHOLD:
        view = ShipMarryView(user_a, user_b, prompt, build_embed, blurb)
    else:
        view = LanguageToggleView(prompt, build_embed)

    sent = await ctx.send(embed=build_embed(blurb), view=view)
    view.message = sent

@bot.command(usage="<user1> [user2]", help="Shows compatibility % between two users (no marry button -- just the vibes).")
async def compatibility(ctx, user1: discord.Member = None, user2: discord.Member = None):
    if user1 is None:
        return await ctx.send("❌ Mention at least one person. Usage: `?compatibility @user` or `?compatibility @user1 @user2`")
    if user2 is None:
        user_a, user_b = ctx.author, user1
    else:
        user_a, user_b = user1, user2

    if user_a.id == user_b.id:
        return await ctx.send("❌ Can't check compatibility with yourself!")

    score = compute_compatibility(user_a.id, user_b.id)
    prompt = build_compatibility_prompt(user_a, user_b, score)
    blurb = await generate_compatibility_blurb(user_a, user_b, score)

    bar_filled = "💗" * (score // 10)
    bar_empty = "🤍" * (10 - score // 10)

    def build_embed(text):
        return discord.Embed(
            title=f"💞 {user_a.display_name} & {user_b.display_name}",
            description=f"{bar_filled}{bar_empty}\n**{score}% Compatible**\n\n*{text}*",
            color=discord.Color.purple()
        )

    view = LanguageToggleView(prompt, build_embed)
    sent = await ctx.send(embed=build_embed(blurb), view=view)
    view.message = sent

@bot.command(help="Propose marriage to another member (they must accept).")
async def marry(ctx, member: discord.Member = None):
    if member is None or member.bot:
        return await ctx.send("❌ You need to mention a real member to marry. Usage: `?marry @user`")
    if member.id == ctx.author.id:
        return await ctx.send("❌ You can't marry yourself!")

    guild_marriages = get_guild_marriages(ctx.guild.id)
    if str(ctx.author.id) in guild_marriages:
        return await ctx.send(f"❌ You're already married on this server! Use `?divorce` first if you want to marry someone else.")
    if str(member.id) in guild_marriages:
        return await ctx.send(f"❌ {member.display_name} is already married to someone else on this server.")

    view = ConfirmRelationshipView(ctx.author, member, "marry")
    view.message = await ctx.send(f"💍 {member.mention}, {ctx.author.mention} wants to marry you! Do you accept?", view=view)

@bot.command(help="Divorces your current partner on this server.")
async def divorce(ctx):
    guild_marriages = get_guild_marriages(ctx.guild.id)
    uid = str(ctx.author.id)
    if uid not in guild_marriages:
        return await ctx.send("❌ You're not currently married on this server.")

    partner_id = guild_marriages.pop(uid)
    guild_marriages.pop(partner_id, None)
    save_data(marriages, MARRIAGES_FILE)

    partner = ctx.guild.get_member(int(partner_id))
    partner_mention = partner.mention if partner else f"<@{partner_id}>"
    await ctx.send(f"💔 {ctx.author.mention} and {partner_mention} are now divorced.")

@bot.command(name="marriages", aliases=["marriedmost"], help="This server's all-time leaderboard of who's been married the most times (NOT current status -- use ?couple for that).")
async def marriage_leaderboard(ctx):
    guild_marriage_stats = get_guild_marriage_stats(ctx.guild.id)
    if not guild_marriage_stats:
        return await ctx.send("💔 Nobody's been married yet on this server.")

    sorted_stats = sorted(guild_marriage_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    trophies = {0: "🥇", 1: "🥈", 2: "🥉"}
    lines = []
    for i, (uid, count) in enumerate(sorted_stats):
        member = ctx.guild.get_member(int(uid))
        name = member.display_name if member else f"User {uid}"
        lines.append(f"{trophies.get(i, f'#{i + 1}')} **{name}**: married {count} time{'s' if count != 1 else ''}")

    embed = discord.Embed(title="💍 Most Married (All-Time, This Server)", description="\n".join(lines), color=discord.Color.pink())
    embed.set_footer(text="This counts total marriages ever on this server, including ones that ended in divorce. Use ?couple @user to check someone's CURRENT status.")
    await ctx.send(embed=embed)

@bot.command(help="Shows whether a user is currently married on this server, and to whom.")
async def couple(ctx, member: discord.Member = None):
    member = member or ctx.author
    uid = str(member.id)
    guild_marriages = get_guild_marriages(ctx.guild.id)

    if uid not in guild_marriages:
        return await ctx.send(f"💔 {member.display_name} is not currently married to anyone on this server.")

    partner_id = guild_marriages[uid]
    partner = ctx.guild.get_member(int(partner_id))
    partner_mention = partner.mention if partner else f"<@{partner_id}>"
    await ctx.send(f"💍 {member.mention} is currently married to {partner_mention}.")

@bot.command(help="Send a friend request to another member (they must accept).")
async def friend(ctx, member: discord.Member = None):
    if member is None or member.bot:
        return await ctx.send("❌ You need to mention a real member to friend. Usage: `?friend @user`")
    if member.id == ctx.author.id:
        return await ctx.send("❌ You can't friend yourself!")

    guild_friends = get_guild_friends(ctx.guild.id)
    if member.id in [int(f) for f in guild_friends.get(str(ctx.author.id), [])]:
        return await ctx.send(f"❌ You're already friends with {member.display_name} on this server.")

    view = ConfirmRelationshipView(ctx.author, member, "friend")
    view.message = await ctx.send(f"🤝 {member.mention}, {ctx.author.mention} wants to be friends! Do you accept?", view=view)

@bot.command(help="Removes a friend from your friends list on this server.")
async def unfriend(ctx, member: discord.Member = None):
    if member is None:
        return await ctx.send("❌ Usage: `?unfriend @user`")

    guild_friends = get_guild_friends(ctx.guild.id)
    uid, tid = str(ctx.author.id), str(member.id)
    if tid not in guild_friends.get(uid, []):
        return await ctx.send(f"❌ You're not friends with {member.display_name} on this server.")

    guild_friends[uid].remove(tid)
    if uid in guild_friends.get(tid, []):
        guild_friends[tid].remove(uid)
    save_data(friends, FRIENDS_FILE)
    await ctx.send(f"💔 {ctx.author.mention} and {member.mention} are no longer friends.")

@bot.command(name="friendslist", aliases=["friends"], help="Shows a user's friends list on this server.")
async def friends_list(ctx, member: discord.Member = None):
    member = member or ctx.author
    guild_friends = get_guild_friends(ctx.guild.id)
    friend_ids = guild_friends.get(str(member.id), [])

    if not friend_ids:
        return await ctx.send(f"📭 {member.display_name} doesn't have any friends registered on this server yet.")

    names = []
    for fid in friend_ids:
        f_member = ctx.guild.get_member(int(fid))
        names.append(f_member.mention if f_member else f"<@{fid}>")

    embed = discord.Embed(
        title=f"🤝 {member.display_name}'s Friends ({len(names)})",
        description="\n".join(names),
        color=discord.Color.green()
    )
    await ctx.send(embed=embed)

# --------------------------------------------------------
# 🔨 UTILITY COMMANDS
# --------------------------------------------------------

@bot.command(help="Checks the bot's latency to the server.")
async def ping(ctx):
    await ctx.send(f'🏓 Pong! Latency is **{round(bot.latency * 1000)}ms**.')

@bot.command(name="gl", aliases=["invite", "invitelink"], help="Gives you a link to add this bot to another server.")
async def get_invite_link(ctx):
    permissions = discord.Permissions(
        view_channel=True, send_messages=True, send_messages_in_threads=True,
        create_public_threads=True, embed_links=True, attach_files=True,
        read_message_history=True, add_reactions=True, manage_messages=True,
        manage_roles=True, manage_nicknames=True, manage_channels=True,
        manage_emojis_and_stickers=True, kick_members=True, ban_members=True,
        moderate_members=True, mention_everyone=True, view_audit_log=True,
    )
    invite_url = discord.utils.oauth_url(bot.application_id, permissions=permissions, scopes=("bot", "applications.commands"))
    embed = discord.Embed(
        title=f"🔗 Add {bot.user.name} to your server!",
        description=f"[**Click here to invite {bot.user.name}**]({invite_url})",
        color=discord.Color.blurple()
    )
    embed.set_thumbnail(url=bot.user.display_avatar.url)
    await ctx.send(embed=embed)

@bot.command(name="setchannel", usage="<birthday|leaderboard|suggestions> [#channel]", help="[Mods only] Sets which channel a feature posts to on THIS server.")
@commands.has_permissions(manage_guild=True)
async def set_feature_channel(ctx, feature: str, channel: discord.TextChannel = None):
    feature = feature.lower()
    key_map = {
        "birthday": "birthday_channel_id", "leaderboard": "leaderboard_channel_id",
        "suggestions": "suggestion_channel_id", "suggestion": "suggestion_channel_id",
        "welcome": "welcome_channel_id", "modlog": "mod_log_channel_id", "mod_log": "mod_log_channel_id",
        "ghostlog": "ghost_log_channel_id", "ghost_log": "ghost_log_channel_id",
        "eulogy": "eulogy_channel_id", "confessions": "confessions_channel_id", "confession": "confessions_channel_id",
    }
    if feature not in key_map:
        return await ctx.send(
            "❌ Feature must be one of: `birthday`, `leaderboard`, `suggestions`, `welcome`, `modlog`, `ghostlog`, `eulogy`, `confessions`.\n"
            "Usage: `?setchannel birthday #general` (or just `?setchannel birthday` while sitting in the target channel)."
        )
    target_channel = channel or ctx.channel
    gid = str(ctx.guild.id)
    if gid not in server_config:
        server_config[gid] = {}
    server_config[gid][key_map[feature]] = str(target_channel.id)
    save_data(server_config, SERVER_CONFIG_FILE)
    await ctx.send(f"✅ **{feature.capitalize()}** messages will now post in {target_channel.mention} on this server.")
    await send_mod_log(ctx.guild, "Server Config Changed", f"**Feature:** {feature}\n**Channel:** {target_channel.mention}", ctx.author)

# --------------------------------------------------------
# 🗣️ ?talk CHANNEL RESTRICTION (Mods only)
# --------------------------------------------------------
@bot.command(name="restricttalk", usage="#channel", help="[Mods only] Whitelists a channel where ?talk is allowed. Once any channel is whitelisted, only whitelisted channels can use ?talk.")
@commands.has_permissions(manage_guild=True)
async def restrict_talk(ctx, channel: discord.TextChannel):
    gid = str(ctx.guild.id)
    server_config.setdefault(gid, {}).setdefault("talk_allowed_channel_ids", [])
    if str(channel.id) in server_config[gid]["talk_allowed_channel_ids"]:
        return await ctx.send(f"❌ {channel.mention} is already whitelisted for `?talk`.")
    server_config[gid]["talk_allowed_channel_ids"].append(str(channel.id))
    save_data(server_config, SERVER_CONFIG_FILE)
    await ctx.send(f"✅ `?talk` is now whitelisted in {channel.mention}. Only whitelisted channels can use it from now on.")
    await send_mod_log(ctx.guild, "?talk Channel Whitelisted", f"**Channel:** {channel.mention}", ctx.author)

@bot.command(name="unrestricttalk", usage="#channel", help="[Mods only] Removes a channel from the ?talk whitelist.")
@commands.has_permissions(manage_guild=True)
async def unrestrict_talk(ctx, channel: discord.TextChannel):
    gid = str(ctx.guild.id)
    allowed = server_config.get(gid, {}).get("talk_allowed_channel_ids", [])
    if str(channel.id) not in allowed:
        return await ctx.send(f"❌ {channel.mention} isn't on the `?talk` whitelist.")
    allowed.remove(str(channel.id))
    save_data(server_config, SERVER_CONFIG_FILE)
    note = " No channels are whitelisted now, so `?talk` works everywhere again." if not allowed else ""
    await ctx.send(f"✅ Removed {channel.mention} from the `?talk` whitelist.{note}")
    await send_mod_log(ctx.guild, "?talk Channel Restriction Removed", f"**Channel:** {channel.mention}", ctx.author)

@bot.command(name="listtalkchannels", help="Shows which channels ?talk is restricted to (if any).")
async def list_talk_channels(ctx):
    allowed = server_config.get(str(ctx.guild.id), {}).get("talk_allowed_channel_ids", [])
    if not allowed:
        return await ctx.send("📭 No channels are whitelisted -- `?talk` works everywhere on this server.")
    lines = []
    for cid in allowed:
        ch = ctx.guild.get_channel(int(cid))
        lines.append(ch.mention if ch else f"*(deleted channel {cid})*")
    embed = discord.Embed(title="🗣️ ?talk Allowed Channels", description="\n".join(lines), color=discord.Color.blurple())
    await ctx.send(embed=embed)

# --------------------------------------------------------
# 💀 RANDOM AMBIENT ROAST TOGGLE (Mods only)
# --------------------------------------------------------
@bot.command(name="disableroast", help="[Mods only] Turns off the automatic random roast (posted every ~6 hours) on this server.")
@commands.has_permissions(manage_guild=True)
async def disable_random_roast(ctx):
    gid = str(ctx.guild.id)
    server_config.setdefault(gid, {})
    if server_config[gid].get("random_roast_disabled"):
        return await ctx.send("❌ The random roast is already turned off on this server.")
    server_config[gid]["random_roast_disabled"] = True
    save_data(server_config, SERVER_CONFIG_FILE)
    await ctx.send("🔇 Random roast turned **off** for this server. Use `?enableroast` to turn it back on.")
    await send_mod_log(ctx.guild, "Random Roast Disabled", "The automatic ~6-hourly random roast has been turned off for this server.", ctx.author)

@bot.command(name="enableroast", help="[Mods only] Turns the automatic random roast back on for this server.")
@commands.has_permissions(manage_guild=True)
async def enable_random_roast(ctx):
    gid = str(ctx.guild.id)
    if not server_config.get(gid, {}).get("random_roast_disabled"):
        return await ctx.send("❌ The random roast is already turned on for this server.")
    server_config[gid]["random_roast_disabled"] = False
    save_data(server_config, SERVER_CONFIG_FILE)
    await ctx.send("🔊 Random roast turned back **on** for this server -- expect one every ~6 hours again.")
    await send_mod_log(ctx.guild, "Random Roast Enabled", "The automatic ~6-hourly random roast has been turned back on for this server.", ctx.author)

# --------------------------------------------------------
# 🔒 PRIVATE MEMBER CHANNELS (?createpvtchannel)
# --------------------------------------------------------
MAX_PRIVATE_CHANNELS_PER_USER = 3

@bot.command(name="createpvtchannel", usage="<name>", help="Creates a private channel only you can see (plus the bot) -- not even mods.")
@commands.bot_has_permissions(manage_channels=True, manage_roles=True)
async def create_pvt_channel(ctx, *, name: str):
    owner_id = ctx.author.id
    owned_count = sum(1 for rec in private_channels.values() if rec.get("guild_id") == ctx.guild.id and rec.get("owner_id") == owner_id)
    if owned_count >= MAX_PRIVATE_CHANNELS_PER_USER:
        return await ctx.send(
            f"❌ You already have {MAX_PRIVATE_CHANNELS_PER_USER} private channels on this server. "
            f"Delete one with `?deletepvtchannel` (run from inside it) before making another."
        )
    overwrites = {
        ctx.guild.default_role: discord.PermissionOverwrite(view_channel=False),
        ctx.guild.me: discord.PermissionOverwrite(view_channel=True, send_messages=True, manage_channels=True, manage_permissions=True, read_message_history=True, embed_links=True, attach_files=True),
        ctx.author: discord.PermissionOverwrite(view_channel=True, send_messages=True, read_message_history=True),
    }
    try:
        channel = await ctx.guild.create_text_channel(name=name, overwrites=overwrites, category=ctx.channel.category, reason=f"Private channel created by {ctx.author} via ?createpvtchannel")
    except discord.Forbidden:
        return await ctx.send("❌ I don't have permission to create channels here. I need **Manage Channels** and **Manage Roles/Permissions**.")
    except Exception as e:
        return await ctx.send(f"❌ Couldn't create the channel: {e}")

    private_channels[str(channel.id)] = {"guild_id": ctx.guild.id, "owner_id": owner_id, "view": [], "type": []}
    save_data(private_channels, PRIVATE_CHANNELS_FILE)
    await ctx.send(
        f"🔒 Created {channel.mention} -- right now only you (and I) can see it. "
        f"Use `?allowin @user` inside it to let someone view it (read-only), and `?allowtype @user` to let "
        f"them start typing too. You can remove someone with `?disallowin @user`, and delete the whole "
        f"channel any time with `?deletepvtchannel`, all from inside it.\n"
        f"⚠️ This hides the channel from regular mods, but a true **Administrator** on this server can "
        f"always see every channel -- that's a Discord-wide limit no bot can override."
    )

@bot.command(name="allowin", usage="@user", help="[Run inside your private channel] Lets a user view (read-only) your private channel.")
async def allow_in_pvt_channel(ctx, member: discord.Member):
    record = private_channels.get(str(ctx.channel.id))
    if not record:
        return await ctx.send("❌ This isn't a private channel created with `?createpvtchannel`.")
    if record["owner_id"] != ctx.author.id:
        return await ctx.send("❌ Only the person who created this channel can do that.")
    if member.id == ctx.author.id:
        return await ctx.send("❌ You already own this channel!")
    try:
        await ctx.channel.set_permissions(member, view_channel=True, send_messages=False, read_message_history=True, reason=f"Allowed in by owner {ctx.author}")
    except discord.Forbidden:
        return await ctx.send("❌ I don't have permission to edit this channel's permissions.")
    if member.id not in record["view"]:
        record["view"].append(member.id)
    save_data(private_channels, PRIVATE_CHANNELS_FILE)
    await ctx.send(f"👀 {member.mention} can now view this channel (read-only, can't type yet). Use `?allowtype @user` to let them type too.")

@bot.command(name="allowtype", usage="@user", help="[Run inside your private channel] Lets a user who's already allowed in start typing.")
async def allow_type_pvt_channel(ctx, member: discord.Member):
    record = private_channels.get(str(ctx.channel.id))
    if not record:
        return await ctx.send("❌ This isn't a private channel created with `?createpvtchannel`.")
    if record["owner_id"] != ctx.author.id:
        return await ctx.send("❌ Only the person who created this channel can do that.")
    try:
        await ctx.channel.set_permissions(member, view_channel=True, send_messages=True, read_message_history=True, reason=f"Allowed to type by owner {ctx.author}")
    except discord.Forbidden:
        return await ctx.send("❌ I don't have permission to edit this channel's permissions.")
    if member.id not in record["view"]:
        record["view"].append(member.id)
    if member.id not in record["type"]:
        record["type"].append(member.id)
    save_data(private_channels, PRIVATE_CHANNELS_FILE)
    await ctx.send(f"⌨️ {member.mention} can now type in this channel.")

@bot.command(name="disallowin", usage="@user", help="[Run inside your private channel] Removes a user's access to your private channel entirely.")
async def disallow_in_pvt_channel(ctx, member: discord.Member):
    record = private_channels.get(str(ctx.channel.id))
    if not record:
        return await ctx.send("❌ This isn't a private channel created with `?createpvtchannel`.")
    if record["owner_id"] != ctx.author.id:
        return await ctx.send("❌ Only the person who created this channel can do that.")
    try:
        await ctx.channel.set_permissions(member, overwrite=None, reason=f"Removed by owner {ctx.author}")
    except discord.Forbidden:
        return await ctx.send("❌ I don't have permission to edit this channel's permissions.")
    record["view"] = [u for u in record["view"] if u != member.id]
    record["type"] = [u for u in record["type"] if u != member.id]
    save_data(private_channels, PRIVATE_CHANNELS_FILE)
    await ctx.send(f"🚫 Removed {member.mention}'s access to this channel.")

@bot.command(name="deletepvtchannel", help="[Run inside your private channel] Deletes your private channel.")
async def delete_pvt_channel(ctx):
    record = private_channels.get(str(ctx.channel.id))
    if not record:
        return await ctx.send("❌ This isn't a private channel created with `?createpvtchannel`.")
    if record["owner_id"] != ctx.author.id:
        return await ctx.send("❌ Only the person who created this channel can delete it.")
    channel_id = ctx.channel.id
    try:
        await ctx.channel.delete(reason=f"Private channel deleted by owner {ctx.author}")
    except discord.Forbidden:
        return await ctx.send("❌ I don't have permission to delete this channel.")
    private_channels.pop(str(channel_id), None)
    save_data(private_channels, PRIVATE_CHANNELS_FILE)

async def apply_afk(user_id: int, guild: Optional[discord.Guild], reason: str) -> str:
    now_iso = datetime.now(timezone.utc).isoformat()
    if guild is not None:
        afk_key = f"{guild.id}_{user_id}"
        afk_users[afk_key] = {'reason': reason, 'time': now_iso}
        save_data(afk_users, AFK_FILE)
        return f"😴 You're now AFK on **{guild.name}**: **{reason}**."
    mutual_guilds = [g for g in bot.guilds if g.get_member(user_id)]
    if not mutual_guilds:
        return "❌ I don't share any servers with you, so there's nowhere to mark you AFK."
    for g in mutual_guilds:
        afk_users[f"{g.id}_{user_id}"] = {'reason': reason, 'time': now_iso}
    save_data(afk_users, AFK_FILE)
    return f"😴 You're now AFK on all {len(mutual_guilds)} server(s) we share: **{reason}**."

@bot.command(help="Sets your status to AFK. Works in a server (that server only) or in DM with the bot (applies across every mutual server).")
async def afk(ctx, *, reason: str = "No reason provided"):
    message = await apply_afk(ctx.author.id, ctx.guild, reason)
    if ctx.guild is not None:
        await ctx.send(message.replace("You're", f"{ctx.author.mention} is", 1) if message.startswith("😴") else message)
    else:
        await ctx.author.send(message)

@bot.tree.command(name="afk", description="Set yourself AFK")
@app_commands.describe(reason="Why you're AFK (optional)")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def afk_slash(interaction: discord.Interaction, reason: str = "No reason provided"):
    message = await apply_afk(interaction.user.id, interaction.guild, reason)
    await interaction.response.send_message(message)

async def build_avatar_embed(requester_display_name: str, member) -> tuple:
    override = custom_avatars.get(str(member.id))
    if override and os.path.exists(override["path"]):
        embed = discord.Embed(title=f"Avatar for {member.display_name}", color=discord.Color.blue())
        file = discord.File(override["path"], filename="avatar.png")
        embed.set_image(url="attachment://avatar.png")
        embed.set_footer(text=f"Requested by {requester_display_name} • Custom avatar set by a moderator")
        return embed, file
    embed = discord.Embed(title=f"Avatar for {member.display_name}", color=discord.Color.blue())
    embed.set_image(url=member.display_avatar.url)
    embed.set_footer(text=f"Requested by {requester_display_name}")
    return embed, None

@bot.command(help="Shows the avatar of a user. Mods can override this with ?addav. Works in DM too.")
async def av(ctx, member: Optional[discord.User]):
    member = member or ctx.author
    embed, file = await build_avatar_embed(ctx.author.display_name, member)
    if file:
        await ctx.send(embed=embed, file=file)
    else:
        await ctx.send(embed=embed)

@bot.tree.command(name="av", description="Shows your avatar, or another user's")
@app_commands.describe(user="Whose avatar to show (optional, defaults to you)")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def av_slash(interaction: discord.Interaction, user: Optional[discord.User] = None):
    member = user or interaction.user
    embed, file = await build_avatar_embed(interaction.user.display_name, member)
    if file:
        await interaction.response.send_message(embed=embed, file=file)
    else:
        await interaction.response.send_message(embed=embed)

# NEW: separate SERVER vs GLOBAL avatar lookups. ?av (above) shows whichever
# one Discord would actually display -- the per-server Nitro avatar if the
# user has set one here, else their global avatar. These two split that
# apart so you can specifically see one or the other.
@bot.command(name="avglobal", aliases=["gav"], help="Shows a user's GLOBAL Discord avatar, ignoring any per-server (Nitro) avatar they've set on this server.")
async def avatar_global(ctx, member: Optional[discord.User]):
    member = member or ctx.author
    override = custom_avatars.get(str(member.id))
    if override and os.path.exists(override["path"]):
        embed = discord.Embed(title=f"Global Avatar for {member.name}", color=discord.Color.blue())
        file = discord.File(override["path"], filename="avatar.png")
        embed.set_image(url="attachment://avatar.png")
        embed.set_footer(text=f"Requested by {ctx.author.display_name} • Custom avatar set by a moderator")
        return await ctx.send(embed=embed, file=file)

    try:
        user_obj = await bot.fetch_user(member.id)  # fetch_user always returns the account-level User, never a per-guild avatar
    except Exception:
        user_obj = member

    embed = discord.Embed(title=f"Global Avatar for {user_obj.name}", color=discord.Color.blue())
    embed.set_image(url=user_obj.display_avatar.url)
    embed.set_footer(text=f"Requested by {ctx.author.display_name}")
    await ctx.send(embed=embed)

@bot.command(name="avserver", aliases=["sav"], help="Shows a user's SERVER-SPECIFIC (Nitro) avatar for THIS server, if they've set one -- tells you if they haven't.")
@commands.guild_only()
async def avatar_server(ctx, member: discord.Member = None):
    member = member or ctx.author
    # BUG FIX: was checking member.avatar, which discord.py actually flattens
    # through to the GLOBAL account avatar (via the @flatten_user decorator
    # on Member), not the per-guild one -- so this always used the wrong
    # value. Member.guild_avatar is the real per-server avatar asset (None if
    # they haven't set one here).
    if member.guild_avatar is None:
        return await ctx.send(
            f"❌ {member.display_name} doesn't have a separate avatar set for **{ctx.guild.name}** -- "
            f"they're just showing their global one. Try `?avglobal` or `?av`."
        )
    embed = discord.Embed(title=f"Server Avatar for {member.display_name}", color=discord.Color.blue())
    embed.set_image(url=member.guild_avatar.url)
    embed.set_footer(text=f"Requested by {ctx.author.display_name} • Specific to {ctx.guild.name}")
    await ctx.send(embed=embed)

@bot.command(name="admindms", help="Shows which commands work in DM with the bot.")
async def admin_dms_info(ctx):
    embed = discord.Embed(
        title="📬 Commands That Work in DM",
        description=(
            "Most commands need a server, but these work directly in DM with me, with anyone.\n\n"
            "`/av`, `/rate`, and `/afk` are also available as **slash commands** -- if User Install is "
            "enabled for this bot in its Developer Portal, those even work inside a DM between you and "
            "someone else (not just DMs with me), since slash commands from a user-installed app don't "
            "need the bot to be a member of the channel at all."
        ),
        color=discord.Color.blurple()
    )
    embed.add_field(name="`?av [user]` / `/av [user]`", value="Shows your avatar, or any user's (by mention/ID) if we share a server.", inline=False)
    embed.add_field(name="`?rate [user]` / `/rate [user]`", value="AI-rates an avatar. Limited to once a day per person -- shared between prefix and slash use, not separate limits.", inline=False)
    embed.add_field(name="`?afk [reason]` / `/afk [reason]`", value="Sets you AFK across every server we both share, all at once.", inline=False)
    await ctx.send(embed=embed)

@bot.command(name="addav", usage="@user (attach an image)", help="[Mods only] Sets a custom avatar override for a user -- shows on ?av everywhere this bot is, until a mod removes it.")
@commands.has_permissions(manage_messages=True)
async def add_custom_avatar(ctx, member: discord.Member):
    if not ctx.message.attachments:
        return await ctx.send("❌ Attach an image with this command. Usage: `?addav @user` + attach an image.")
    attachment = ctx.message.attachments[0]
    if not (attachment.content_type and attachment.content_type.startswith("image/")):
        return await ctx.send("❌ That attachment isn't an image.")
    ext = os.path.splitext(attachment.filename)[1] or ".png"
    save_path = os.path.join(CUSTOM_AVATARS_DIR, f"{member.id}{ext}")
    old_override = custom_avatars.get(str(member.id))
    if old_override and old_override.get("path") and old_override["path"] != save_path and os.path.exists(old_override["path"]):
        try:
            os.remove(old_override["path"])
        except Exception as e:
            print(f"⚠️ Failed to remove old custom avatar file: {e}")
    try:
        await attachment.save(save_path)
    except Exception as e:
        return await ctx.send(f"❌ Failed to save that image: {e}")
    custom_avatars[str(member.id)] = {"path": save_path, "set_by": ctx.author.id, "set_at": datetime.now(timezone.utc).isoformat()}
    save_data(custom_avatars, CUSTOM_AVATARS_FILE)
    await ctx.send(f"✅ Custom avatar set for {member.mention}. Running `?av` on them will now show this image everywhere this bot is added, until a mod removes it with `?removeav @user`.")
    await send_mod_log(ctx.guild, "Custom Avatar Set", f"**User:** {member.mention}", ctx.author)

@bot.command(name="removeav", usage="@user", help="[Mods only] Removes a user's custom avatar override, back to their real Discord avatar.")
@commands.has_permissions(manage_messages=True)
async def remove_custom_avatar(ctx, member: discord.Member):
    override = custom_avatars.get(str(member.id))
    if not override:
        return await ctx.send(f"❌ {member.display_name} doesn't have a custom avatar override set.")
    if override.get("path") and os.path.exists(override["path"]):
        try:
            os.remove(override["path"])
        except Exception as e:
            print(f"⚠️ Failed to remove custom avatar file: {e}")
    custom_avatars.pop(str(member.id), None)
    save_data(custom_avatars, CUSTOM_AVATARS_FILE)
    await ctx.send(f"✅ Removed {member.mention}'s custom avatar override. `?av` will show their real Discord avatar again.")
    await send_mod_log(ctx.guild, "Custom Avatar Removed", f"**User:** {member.mention}", ctx.author)

@bot.command(help="Shows info about a user.")
async def userinfo(ctx, member: Optional[discord.Member]):
    member = member or ctx.author
    embed = discord.Embed(title=f"User Info: {member.display_name}", description=member.mention, color=member.color)
    embed.set_thumbnail(url=member.display_avatar.url)
    join_delta = datetime.now(timezone.utc) - member.joined_at.replace(tzinfo=timezone.utc)
    join_days = join_delta.days
    creation_delta = datetime.now(timezone.utc) - member.created_at.replace(tzinfo=timezone.utc)
    creation_days = creation_delta.days
    embed.add_field(name="ID", value=member.id, inline=False)
    embed.add_field(name="Joined Server", value=f"{member.joined_at.strftime('%Y-%m-%d')} ({join_days} days ago)", inline=False)
    embed.add_field(name="Account Created", value=f"{member.created_at.strftime('%Y-%m-%d')} ({creation_days} days ago)", inline=False)
    gid = str(ctx.guild.id)
    uid = str(member.id)
    record = member_history.get(gid, {}).get(uid)
    if record:
        if "first_joined" in record:
            first_joined_dt = datetime.fromisoformat(record["first_joined"])
            embed.add_field(name="First Joined", value=first_joined_dt.strftime('%Y-%m-%d %H:%M UTC'), inline=True)
        join_count = record.get("join_count", 1)
        if join_count > 1:
            embed.add_field(name="Times Joined", value=f"{join_count}", inline=True)
        if record.get("last_left"):
            last_left_dt = datetime.fromisoformat(record["last_left"])
            embed.add_field(name="Last Left Server", value=last_left_dt.strftime('%Y-%m-%d %H:%M UTC'), inline=True)
    roles = [role.name for role in member.roles if role.name != '@everyone']
    if roles:
        embed.add_field(name=f"Roles ({len(roles)})", value=", ".join(roles), inline=False)
    await ctx.send(embed=embed)

@bot.command(help="Makes the bot say a message and deletes the original command.")
@commands.has_permissions(manage_messages=True)
async def say(ctx, *, message: str):
    await ctx.message.delete()
    await ctx.send(message)
    await send_mod_log(ctx.guild, "?say Used", f"**Channel:** {ctx.channel.mention}\n**Message:** {message}", ctx.author)

@bot.command()
async def online(ctx):
    if ctx.author.name.lower() not in STATUS_COMMAND_OWNERS:
        return await ctx.send("❌ Only **kanjuubarfiiii** or **huh.ashh** can do this!")
    global current_bot_status
    current_bot_status = discord.Status.online
    await bot.change_presence(status=current_bot_status, activity=current_bot_activity)
    await ctx.send("🟢 Status changed to **Online**.")

@bot.command()
async def dnd(ctx):
    if ctx.author.name.lower() not in STATUS_COMMAND_OWNERS:
        return await ctx.send("❌ Only **kanjuubarfiiii** or **huh.ashh** can do this!")
    global current_bot_status
    current_bot_status = discord.Status.do_not_disturb
    await bot.change_presence(status=current_bot_status, activity=current_bot_activity)
    await ctx.send("🔴 Status changed to **Do Not Disturb**.")

@bot.command()
async def idle(ctx):
    if ctx.author.name.lower() not in STATUS_COMMAND_OWNERS:
        return await ctx.send("❌ Only **kanjuubarfiiii** or **huh.ashh** can do this!")
    global current_bot_status
    current_bot_status = discord.Status.idle
    await bot.change_presence(status=current_bot_status, activity=current_bot_activity)
    await ctx.send("🌙 Status changed to **Idle**.")

CHATSBOT_ALLOWED_USERNAMES = {"kanjuubarfiiii"}

@bot.command(name="chatsbot", usage="@user", help="[Owner only] Dumps a transcript of a user's recent ?talk conversation with the bot.")
async def chats_bot(ctx, user: discord.User):
    if ctx.author.name.lower() not in CHATSBOT_ALLOWED_USERNAMES:
        return await ctx.send("❌ You're not permitted to use this command.")
    history = user_chats.get(user.id)
    if not history:
        return await ctx.send(f"📭 No stored `?talk` conversation history for {user.name}.")
    lines = []
    for entry in history:
        speaker = "**User:**" if entry["role"] == "user" else "**Bot:**"
        lines.append(f"{speaker} {entry['content']}")
    transcript = "\n\n".join(lines)
    header = f"📜 **?talk transcript for {user.name}** ({user.id}) -- last {len(history)} messages:\n\n"
    await safe_send(ctx.channel, header + transcript, reply_to=ctx.message)

@bot.command(name="botsetup", help="[Owner only] Shows the setup checklist for adding this bot to a new server.")
async def bot_setup_guide(ctx):
    if ctx.author.name.lower() not in CHATSBOT_ALLOWED_USERNAMES:
        return await ctx.send("❌ You're not permitted to use this command.")
    embed1 = discord.Embed(title="🛠️ New Server Setup Checklist (1/2)", description="Run these once, in the new server, right after inviting the bot with `?gl`.", color=discord.Color.blurple())
    embed1.add_field(name="📌 Channels to create (or reuse existing ones)", value=("• A **welcome** channel\n• A **mod-log** channel (named exactly `mod-log` works automatically, or configure any name via `?setchannel modlog`)\n• A **ghost-ping log** channel (mods only)\n• A **general/leaderboard** channel\n• A **birthday announcements** channel\n• A **suggestions** channel\n• An **eulogy** channel (optional)\n• A **confessions** channel (optional)"), inline=False)
    embed1.add_field(name="⚙️ Then run `?setchannel <type> #channel` for each:", value="`welcome` `modlog` `ghostlog` `leaderboard` `birthday` `suggestions` `eulogy` `confessions`", inline=False)
    await ctx.send(embed=embed1)
    embed2 = discord.Embed(title="🛠️ New Server Setup Checklist (2/2)", color=discord.Color.blurple())
    embed2.add_field(name="🎭 Self-roles", value="Run `?setup_roles` in whichever channel should host the role-picker menu.", inline=False)
    embed2.add_field(name="🗣️ ?talk restriction (optional)", value="If `?talk` should only work in specific channels, run `?restricttalk #channel` for each. Skip to leave it open everywhere.", inline=False)
    embed2.add_field(name="🔑 Environment variables to double-check on the host", value="`DISCORD_TOKEN`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `DATA_DIR` (for persistent storage across redeploys), optionally `GIPHY_API_KEY`.", inline=False)
    embed2.add_field(name="✅ Required bot permissions", value="Covered automatically by the `?gl` invite link -- Manage Roles, Manage Channels, Manage Messages, Kick/Ban Members, Moderate Members, View Audit Log, and the usual message/embed/reaction permissions.", inline=False)
    await ctx.send(embed=embed2)

# --------------------------------------------------------
# 🤖 BOT IDENTITY COMMANDS (Admins only)
# --------------------------------------------------------

@bot.command(name="setpfp", usage="<image URL, or attach a file>", help="[Admins only] Changes the bot's avatar globally (every server it's in).")
@commands.has_permissions(administrator=True)
async def set_bot_avatar(ctx, url: str = None):
    image_bytes = None
    if ctx.message.attachments:
        image_bytes = await ctx.message.attachments[0].read()
    elif url:
        try:
            async with bot.http_session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    image_bytes = await resp.read()
                else:
                    return await ctx.send(f"❌ Couldn't fetch that image (status {resp.status}).")
        except Exception as e:
            return await ctx.send(f"❌ Failed to fetch that image: {e}")
    else:
        return await ctx.send("❌ Attach an image, or use `?setpfp <image URL>`.")
    try:
        await bot.user.edit(avatar=image_bytes)
        await ctx.send("✅ Bot avatar updated! (Heads up: this changes it everywhere the bot is added, not just this server.)")
    except discord.HTTPException as e:
        await ctx.send(f"❌ Discord rejected the avatar change: {e}. Note: avatar changes are rate-limited to roughly twice per hour.")
    except Exception as e:
        await ctx.send(f"❌ Failed to update avatar: {e}")

@bot.command(name="setactivity", usage="<playing|watching|listening|competing> <text>", help="[Admins only] Changes the bot's presence activity text (e.g. 'Playing ?help').")
@commands.has_permissions(administrator=True)
async def set_bot_activity(ctx, activity_type: str, *, text: str):
    activity_type = activity_type.lower()
    type_map = {"playing": discord.ActivityType.playing, "watching": discord.ActivityType.watching, "listening": discord.ActivityType.listening, "competing": discord.ActivityType.competing}
    if activity_type not in type_map:
        return await ctx.send("❌ Activity type must be one of: `playing`, `watching`, `listening`, `competing`.\nUsage: `?setactivity watching the server`")
    global current_bot_activity
    current_bot_activity = discord.Activity(type=type_map[activity_type], name=text)
    await bot.change_presence(status=current_bot_status, activity=current_bot_activity)
    await ctx.send(f"✅ Bot activity updated to: **{activity_type.capitalize()} {text}**")

@bot.command(name="setabout", usage="<text>", help="[Admins only] Changes the bot's 'About Me' description shown on its profile.")
@commands.has_permissions(administrator=True)
async def set_bot_about(ctx, *, text: str):
    if len(text) > 400:
        return await ctx.send(f"❌ Discord limits the About Me description to 400 characters (yours is {len(text)}).")
    try:
        async with bot.http_session.patch(
            "https://discord.com/api/v10/applications/@me",
            headers={"Authorization": f"Bot {os.getenv('DISCORD_TOKEN')}", "Content-Type": "application/json"},
            json={"description": text}, timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status == 200:
                await ctx.send("✅ Bot's About Me description updated! (This is also global -- shows on its profile everywhere.)")
            else:
                body = await resp.text()
                await ctx.send(f"❌ Discord rejected the update (status {resp.status}): {body[:200]}")
    except Exception as e:
        await ctx.send(f"❌ Failed to update About Me: {e}")

@bot.command(name="afkclear")
@commands.has_permissions(manage_messages=True)
async def afk_clear(ctx, member: discord.Member):
    afk_key = f"{ctx.guild.id}_{member.id}"
    if afk_key in afk_users:
        del afk_users[afk_key]
        save_data(afk_users, AFK_FILE)
        await ctx.send(f"✅ AFK status for {member.display_name} has been cleared by a moderator.")
        await send_mod_log(ctx.guild, "AFK Force-Cleared", f"**User:** {member.mention}", ctx.author)
    else:
        await ctx.send(f"❌ {member.display_name} is not currently AFK.")

#------Banner------

@bot.command(name="banner", help="Get a user's profile banner.")
async def banner(ctx, member: discord.Member = None):
    member = member or ctx.author
    user = await bot.fetch_user(member.id)
    if user.banner:
        banner_url = user.banner.url
        embed = discord.Embed(title=f"🖼️ {member.display_name}'s Banner", color=member.color)
        embed.set_image(url=banner_url)
        await ctx.send(embed=embed)
    else:
        if user.accent_color:
            await ctx.send(f"❌ {member.display_name} doesn't have a banner image, but their profile accent color is `{user.accent_color}`.")
        else:
            await ctx.send(f"❌ {member.display_name} does not have a banner.")

# --- Scheduled Command ---
@bot.command()
@commands.has_permissions(manage_messages=True)
async def schedule(ctx, movie: str, date: str, time: str):
    """Schedules an announcement for a movie or event. Use MM/DD/YYYY HH:MM."""
    try:
        scheduled_datetime_naive = datetime.strptime(f"{date} {time}", "%m/%d/%Y %H:%M")
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        scheduled_datetime_aware = scheduled_datetime_naive.replace(tzinfo=ist_tz)
        scheduled_utc = scheduled_datetime_aware.astimezone(timezone.utc)
        now_aware = datetime.now(timezone.utc)
        delay = (scheduled_utc - now_aware).total_seconds()
        if delay <= 0:
            await ctx.send("The scheduled time is in the past. Please choose a future time.")
            return
        td = scheduled_utc - now_aware
        total_minutes = td.seconds // 60
        hours = total_minutes // 60
        minutes = total_minutes % 60
        await ctx.send(f"✅ Scheduled movie announcement for **'{movie}'** on **{date} at {time} IST** (in about {hours} hours and {minutes} minutes).")
        await send_mod_log(ctx.guild, "Movie Scheduled", f"{ctx.author.mention} scheduled '{movie}' for {date} {time} IST.", ctx.author)
        await asyncio.sleep(delay)
        channel = ctx.channel
        await channel.send(f"@everyone 🍿 **Movie Time!** 🎬 **'{movie}'** is starting now! Grab your popcorn!")
    except ValueError:
        await ctx.send("⚠️ Invalid date or time format. Use **MM/DD/YYYY** for date and **HH:MM** for time (24-hour, e.g., 21:30).")

# --------------------------------------------------------
# 🎲 DICE ROLL GAME
# --------------------------------------------------------
last_droll_time = {}

@bot.command(name="droll", usage="<count> <min>-<max>", help='Rolls <count> random numbers in a range, then finds whoever called one of those numbers FIRST in recent chat and crowns them the winner. Usage: ?droll 4 1-10')
async def dice_roll(ctx, count: int, number_range: str):
    if count < 1 or count > 20:
        return await ctx.send("❌ Count must be between 1 and 20.")
    try:
        low_str, high_str = number_range.split("-")
        low, high = int(low_str), int(high_str)
    except ValueError:
        return await ctx.send('❌ Range must look like `1-10`. Usage: `?droll 4 1-10`')
    if low >= high:
        return await ctx.send("❌ The first number in the range must be smaller than the second.")
    rolled = [random.randint(low, high) for _ in range(count)]
    rolled_strs = {str(n) for n in rolled}
    search_after = last_droll_time.get(ctx.channel.id)
    async with ctx.typing():
        matches = []
        async for m in ctx.channel.history(limit=500, after=search_after):
            if m.author.bot:
                continue
            if m.content.strip() in rolled_strs:
                matches.append(m)
    last_droll_time[ctx.channel.id] = ctx.message.created_at
    embed = discord.Embed(title=f"🎲 Rolled {count} number{'s' if count != 1 else ''} ({low}-{high})", description=f"**Results:** {', '.join(str(n) for n in rolled)}", color=discord.Color.purple())
    if matches:
        matches.sort(key=lambda m: m.created_at)
        winner_msg = matches[0]
        embed.add_field(name="🏆 Winner", value=f"{winner_msg.author.mention} called **{winner_msg.content.strip()}** first! [Jump]({winner_msg.jump_url})", inline=False)
    else:
        embed.add_field(name="🏆 Winner", value="No one called any of these numbers since the last roll in this channel!", inline=False)
    await ctx.send(embed=embed)

@bot.command(usage='"<question>" <option1> <option2> ...', help='Creates a reaction poll. Usage: ?poll "question" option1 option2 ...')
async def poll(ctx, question: str, *options: str):
    if len(options) < 2:
        return await ctx.send('❌ Provide at least 2 options. Usage: `?poll "question" option1 option2 ...`')
    if len(options) > 10:
        return await ctx.send("❌ Max 10 options allowed.")
    description = "\n".join(f"{NUMBER_EMOJIS[i]} {opt}" for i, opt in enumerate(options))
    embed = discord.Embed(title=f"📊 {question}", description=description, color=discord.Color.blurple())
    embed.set_footer(text=f"Poll by {ctx.author.display_name} -- react below to vote!")
    msg = await ctx.send(embed=embed)
    for i in range(len(options)):
        try:
            await msg.add_reaction(NUMBER_EMOJIS[i])
        except discord.Forbidden:
            print("⚠️ Missing permission to add reactions for ?poll.")
            break

@bot.command(help="Submit a suggestion for the server. Usage: ?suggest <your idea>")
async def suggest(ctx, *, idea: str):
    channel = get_configured_channel(ctx.guild, "suggestion_channel_id") or ctx.channel
    embed = discord.Embed(title="💡 New Suggestion", description=idea, color=discord.Color.blue(), timestamp=datetime.now(timezone.utc))
    embed.set_author(name=ctx.author.display_name, icon_url=ctx.author.display_avatar.url)
    embed.set_footer(text=f"Submitted by {ctx.author.id}")
    try:
        msg = await channel.send(embed=embed)
        await msg.add_reaction("👍")
        await msg.add_reaction("👎")
        if channel.id != ctx.channel.id:
            await ctx.send(f"✅ Your suggestion has been posted in {channel.mention}!")
    except discord.Forbidden:
        await ctx.send("❌ I don't have permission to post in the suggestions channel. A mod should run `?setchannel suggestions #channel` to set it up, or check my permissions there.")

# --------------------------------------------------------
# 🧠 GEMINI AI COMMANDS
# --------------------------------------------------------
@bot.command()
async def talk(ctx, *, query):
    """Fast AI Chat using Groq (remembers context per-user until you use ?reset).
    Tip: just reply directly to the bot's answer to keep chatting without retyping ?talk."""
    if ctx.guild and not is_talk_allowed_in_channel(ctx.guild, ctx.channel.id):
        return await ctx.send("❌ `?talk` isn't allowed in this channel. Check `?listtalkchannels` for where it's permitted.")
    async with ctx.typing():
        response = await asyncio.to_thread(get_groq_chat_response, ctx.author.id, query)
        sent_messages = await safe_send(ctx.channel, response, reply_to=ctx.message)
        for m in sent_messages:
            track_talk_message(m.id, ctx.author.id)


RATE_PROMPT = (
    "You're looking at this Discord user's profile picture. Look at the actual pixels carefully -- "
    "do not guess based on what a 'typical' Discord avatar might be. Reply with exactly these sections:\n"
    "0. **What's literally visible** -- one plain sentence stating only concrete, directly-observable "
    "facts: subject type (real photo / anime-style drawing / 3D render / logo / meme / abstract / etc.), "
    "dominant colors, and rough composition. Do not name anyone yet in this step.\n"
    "1. **Identification** -- Based ONLY on what you described in step 0, figure out what/who is in "
    "the picture. If it's a real celebrity, public figure, athlete, or influencer, name them "
    "specifically. If it's a fictional character (anime, cartoon, video game, movie, comic, meme), "
    "name the character AND the franchise. Only make an identification if the visual details actually "
    "support it -- if you're genuinely not confident, say plainly 'not confident enough to identify' "
    "instead of guessing a name. If it's clearly just an original photo/drawing/abstract image with no "
    "recognizable character, say that instead.\n"
    "2. **Rating** -- A rating out of 10 (e.g. '7/10'), grounded strictly in what you described in step "
    "0: image quality/resolution, composition, crop, and how well it works as a profile picture. Use "
    "the full 1-10 range -- blurry/low-effort/awkwardly-cropped scores low (2-4), fine-but-generic "
    "scores mid (5-6), genuinely strong and well-composed scores high (8-10).\n"
    "3. **Description** -- 1-2 sentences on what's visible, referencing your step 0 observations.\n"
    "4. **Suggestion** -- One specific, useful tip to improve it.\n"
    "Only report step 0 through 4, be direct and a little witty, and never state something as fact "
    "that you didn't actually observe in the image. Keep the whole reply under 150 words."
)

async def build_rate_response(invoker_id: int, member) -> tuple:
    invoker_str = str(invoker_id)
    last_used_str = rate_usage.get(invoker_str)
    if last_used_str:
        last_used = datetime.fromisoformat(last_used_str)
        elapsed = datetime.now(timezone.utc) - last_used
        if elapsed.total_seconds() < 86400:
            remaining = timedelta(seconds=86400) - elapsed
            hours, remainder = divmod(int(remaining.total_seconds()), 3600)
            minutes = remainder // 60
            return None, f"⏳ You can only use `?rate`/`/rate` once per day. Try again in **{hours}h {minutes}m**."
    try:
        avatar_asset = member.display_avatar.with_size(512).with_format("png")
        avatar_bytes = await avatar_asset.read()
    except Exception as e:
        print(f"🚨 Failed to fetch avatar for rate: {e}")
        return None, "❌ Couldn't fetch that user's avatar."
    try:
        response = await asyncio.to_thread(get_gemini_vision_text, avatar_bytes, "image/png", RATE_PROMPT, 0.5)
    except Exception as e:
        print(f"🚨 Gemini vision error in rate: {e}")
        return None, "❌ Sorry, the rating service is temporarily unavailable. Try again in a bit."
    rate_usage[invoker_str] = datetime.now(timezone.utc).isoformat()
    save_data(rate_usage, RATE_USAGE_FILE)
    embed = discord.Embed(title=f"🖼️ Avatar Rating: {member.name}", description=truncate_text(response, 4000), color=0xFFD700)
    embed.set_thumbnail(url=member.display_avatar.url)
    return embed, None

@bot.command(help="Rates a user's avatar using AI vision. Limited to once per day per person, whether used in a server or DM.")
async def rate(ctx, member: discord.User = None):
    member = member or ctx.author
    embed, error = await build_rate_response(ctx.author.id, member)
    if error:
        await ctx.send(error)
    else:
        await ctx.send(embed=embed)

@bot.tree.command(name="rate", description="AI-rates an avatar (once per day, shared with ?rate)")
@app_commands.describe(user="Whose avatar to rate (optional, defaults to you)")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def rate_slash(interaction: discord.Interaction, user: Optional[discord.User] = None):
    member = user or interaction.user
    await interaction.response.defer()
    embed, error = await build_rate_response(interaction.user.id, member)
    if error:
        await interaction.followup.send(error)
    else:
        await interaction.followup.send(embed=embed)

@bot.command(help="AI reads the recent chat and gives a fun read on the server's current vibe/energy.")
async def vibecheck(ctx):
    async with ctx.typing():
        messages = [m async for m in ctx.channel.history(limit=50) if not m.author.bot and m.content.strip()]
        messages.reverse()
        if len(messages) < 5:
            return await ctx.send("📉 Not enough recent chat activity to vibe check yet -- get talking!")
        transcript = "\n".join(f"{m.author.display_name}: {m.content}" for m in messages[-40:])
        prompt = (
            "Here's a recent chat transcript from a Discord server:\n\n"
            f"{transcript}\n\n"
            "Give a short, funny, punchy 'vibe check' read on the energy/mood of this conversation "
            "(2-4 sentences, with emojis). Don't quote messages directly, just describe the overall vibe."
        )
        try:
            response = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"🚨 Vibecheck error: {e}")
            return await ctx.send("❌ Couldn't generate a vibe check right now.")
        def build_embed(text):
            return discord.Embed(title="📊 Vibe Check", description=text, color=discord.Color.purple())
        view = LanguageToggleView(prompt, build_embed)
        sent = await ctx.send(embed=build_embed(response), view=view)
        view.message = sent

@bot.command(help="AI roasts a user based on their actual server activity. All in good fun!")
async def roast(ctx, member: discord.Member = None):
    member = member or ctx.author
    context = _birthday_context(member, ctx.guild)
    prompt = (
        "Write a short, funny, PG-13 roast (2-4 sentences) of a Discord user based on this context "
        f"about their server activity:\n{context}\n"
        "Keep it playful and lighthearted, not genuinely mean or offensive -- this is for laughs among "
        "friends. Mention their name. Output only the roast, nothing else."
    )
    try:
        response = await asyncio.to_thread(get_groq_text, prompt)
    except Exception as e:
        print(f"🚨 Roast error: {e}")
        return await ctx.send("❌ Couldn't cook up a roast right now.")
    def build_embed(text):
        embed = discord.Embed(title=f"🔥 Roasting {member.display_name}", description=text, color=discord.Color.orange())
        embed.set_thumbnail(url=member.display_avatar.url)
        return embed
    view = LanguageToggleView(prompt, build_embed)
    sent = await ctx.send(embed=build_embed(response), view=view)
    view.message = sent

@bot.command(help="A 'Spotify Wrapped'-style recap of a user's time in the server.")
async def wrapped(ctx, member: discord.Member = None):
    member = member or ctx.author
    gid = str(ctx.guild.id)
    uid = str(member.id)
    msg_count = message_counts.get(gid, {}).get(uid, 0)
    counts = message_counts.get(gid, {})
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    rank = next((i + 1 for i, (u, c) in enumerate(sorted_counts) if u == uid), None)
    record = member_history.get(gid, {}).get(uid, {})
    first_joined = record.get("first_joined")
    join_count = record.get("join_count", 1)
    account_age_days = (datetime.now(timezone.utc) - member.created_at.replace(tzinfo=timezone.utc)).days
    context = _birthday_context(member, ctx.guild)
    prompt = (
        f"Write a fun, punchy one-line 'Spotify Wrapped'-style headline (max 15 words) for this "
        f"Discord user's server activity:\n{context}\nOutput only the headline, nothing else."
    )
    try:
        headline = await asyncio.to_thread(get_groq_text, prompt)
    except Exception:
        headline = f"{member.display_name}'s server story continues..."
    def build_embed(text):
        embed = discord.Embed(title=f"🎁 {member.display_name} Wrapped", description=f"*{text}*", color=discord.Color.magenta())
        embed.add_field(name="💬 Messages This Week", value=str(msg_count), inline=True)
        embed.add_field(name="🏆 Server Rank", value=f"#{rank}" if rank else "Unranked", inline=True)
        embed.add_field(name="🔁 Times Joined", value=str(join_count), inline=True)
        embed.add_field(name="🎂 Account Age", value=f"{account_age_days} days", inline=True)
        embed.set_thumbnail(url=member.display_avatar.url)
        embed.set_footer(text="Note: message stats reset weekly, so this reflects the current week only.")
        return embed
    view = LanguageToggleView(prompt, build_embed)
    sent = await ctx.send(embed=build_embed(headline.strip()), view=view)
    view.message = sent

@bot.command(help="AI summarizes the recent chat in this channel so you can catch up fast.")
async def tldr(ctx, minutes: int = 60):
    if minutes < 1 or minutes > 720:
        return await ctx.send("❌ Please pick a time window between 1 and 720 minutes.")
    async with ctx.typing():
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        messages = [m async for m in ctx.channel.history(after=since, limit=300) if not m.author.bot and m.content.strip()]
        if len(messages) < 3:
            return await ctx.send(f"📭 Not much happened in the last {minutes} minutes to summarize.")
        transcript = "\n".join(f"{m.author.display_name}: {m.content}" for m in messages)
        prompt = (
            f"Summarize this Discord chat transcript from the last {minutes} minutes into a short TL;DR "
            f"(3-6 bullet points):\n\n{transcript}\n\n"
            "Focus on the main topics/decisions/events discussed. Don't quote messages verbatim -- paraphrase."
        )
        try:
            response = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"🚨 TLDR error: {e}")
            return await ctx.send("❌ Couldn't generate a summary right now.")
        def build_embed(text):
            return discord.Embed(title=f"📋 TL;DR -- Last {minutes} Minutes", description=text, color=discord.Color.teal())
        view = LanguageToggleView(prompt, build_embed)
        sent = await ctx.send(embed=build_embed(response), view=view)
        view.message = sent

@bot.command(name="onthisday", help="Resurfaces a random message sent on this exact day in a previous year, in this channel.")
async def on_this_day(ctx):
    async with ctx.typing():
        now = datetime.now(timezone.utc)
        matches = []
        for years_back in range(1, 6):
            try:
                target_date = now.replace(year=now.year - years_back)
            except ValueError:
                continue
            start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            try:
                async for m in ctx.channel.history(after=start, before=end, limit=200):
                    if not m.author.bot and m.content.strip():
                        matches.append(m)
            except Exception as e:
                print(f"⚠️ onthisday history fetch failed for {years_back} years back: {e}")
        if not matches:
            return await ctx.send("📭 Nothing on record for this exact day in previous years -- yet!")
        chosen = random.choice(matches)
        years_ago = now.year - chosen.created_at.year
        embed = discord.Embed(title=f"📅 On This Day ({years_ago} year{'s' if years_ago != 1 else ''} ago)", description=chosen.content, color=discord.Color.blue(), timestamp=chosen.created_at)
        embed.set_author(name=chosen.author.display_name, icon_url=chosen.author.display_avatar.url)
        embed.add_field(name="Jump to Message", value=f"[Click Here]({chosen.jump_url})", inline=False)
        await ctx.send(embed=embed)

@bot.command(help="Shows this server's current 'mood' -- drifts based on recent chat energy.")
async def servermood(ctx):
    gid = str(ctx.guild.id)
    state = server_mood.get(gid, {"score": 0.0})
    score = state["score"]
    label = mood_label_for_score(score)
    prompt = (
        f"A Discord server's current 'mood' is measured at {label} (internal energy score {score:.2f} "
        f"on a scale from -1 quiet to +1 chaotic). Write one short, fun sentence describing this vibe "
        f"like a weather forecaster reporting today's conditions. Output only the sentence."
    )
    try:
        flavor = await asyncio.to_thread(get_groq_text, prompt)
    except Exception:
        flavor = "The vibes are... immeasurable today."
    bar_position = int((score + 1) / 2 * 10)
    bar = "".join("🟪" if i == bar_position else "⬜" for i in range(11))
    embed = discord.Embed(title=f"🎭 Server Mood: {label.split()[0].capitalize()} {label.split()[-1]}", description=f"{bar}\n\n*{flavor}*", color=discord.Color.dark_magenta())
    await ctx.send(embed=embed)

@bot.command(help="AI-generated 'character arc' narrative of a user's time in the server, based on real activity.")
async def evolution(ctx, member: discord.Member = None):
    member = member or ctx.author
    gid, uid = str(ctx.guild.id), str(member.id)
    record = member_history.get(gid, {}).get(uid, {})
    msg_count = message_counts.get(gid, {}).get(uid, 0)
    reps = reputation.get(uid, 0)
    badges = get_earned_badges(member, ctx.guild)
    is_married = uid in get_guild_marriages(ctx.guild.id)
    times_married = get_guild_marriage_stats(ctx.guild.id).get(uid, 0)
    first_joined = record.get("first_joined", "unknown")
    join_count = record.get("join_count", 1)
    context = (
        f"First joined: {first_joined}\n"
        f"Times rejoined: {join_count}\n"
        f"Messages this week: {msg_count}\n"
        f"Reputation: {reps}\n"
        f"Badges earned: {', '.join(b[1] for b in badges) if badges else 'none yet'}\n"
        f"Currently married: {'yes' if is_married else 'no'}\n"
        f"Total times married: {times_married}\n"
    )
    prompt = (
        f"Write a short, dramatic 'character arc' narrative (4-6 sentences) for a Discord user named "
        f"{member.display_name}, framed like a movie synopsis, based on this real data about them:\n{context}\n"
        "Make it feel like an epic personal journey, playful and a little over-the-top, but grounded in "
        "the actual facts given. Output only the narrative."
    )
    async with ctx.typing():
        try:
            narrative = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"⚠️ Evolution generation failed: {e}")
            narrative = "Their story is still being written..."
    def build_embed(text):
        embed = discord.Embed(title=f"🎬 The Evolution of {member.display_name}", description=text, color=discord.Color.dark_gold())
        embed.set_thumbnail(url=member.display_avatar.url)
        return embed
    view = LanguageToggleView(prompt, build_embed)
    sent = await ctx.send(embed=build_embed(narrative), view=view)
    view.message = sent

@bot.command(help="AI writes a fake 'wiki page' history of this server based on real stats.")
async def serverlore(ctx):
    guild = ctx.guild
    gid = str(guild.id)
    counts = message_counts.get(gid, {})
    top_user_id, top_count = max(counts.items(), key=lambda x: x[1], default=(None, 0))
    top_member = guild.get_member(int(top_user_id)) if top_user_id else None
    total_warnings = sum(len(w) for w in warnings_data.get(gid, {}).values())
    total_marriages_ever = sum(marriage_stats.get(gid, {}).values())
    total_memories_saved = len(memory_bank.get(gid, []))
    total_triggers = len(triggers.get(gid, {}))
    earliest_message_note = "lost to time"
    channel = get_configured_channel(guild, "leaderboard_channel_id", LEADERBOARD_CHANNEL_ID, LEADERBOARD_CHANNEL_NAME)
    if channel:
        try:
            async for m in channel.history(limit=1, oldest_first=True):
                earliest_message_note = f'"{m.content[:80]}" -- {m.author.display_name}' if m.content else f"an attachment from {m.author.display_name}"
        except Exception:
            pass
    context = (
        f"Server name: {guild.name}\n"
        f"Server created: {guild.created_at.strftime('%Y-%m-%d')}\n"
        f"Member count: {guild.member_count}\n"
        f"Most active chatter this week: {top_member.display_name if top_member else 'unknown'} ({top_count} messages)\n"
        f"Total warnings issued (all-time): {total_warnings}\n"
        f"Total marriages performed (all-time): {total_marriages_ever}\n"
        f"Memories saved to the memory bank: {total_memories_saved}\n"
        f"Custom triggers configured: {total_triggers}\n"
        f"Earliest known message on record: {earliest_message_note}\n"
    )
    prompt = (
        f"Write a fake, funny 'wiki page' / mockumentary-style history writeup of this Discord server, "
        f"based on these REAL stats:\n{context}\n"
        "Style it like an over-the-top Wikipedia intro paragraph or documentary narrator. Keep it to "
        "6-8 sentences. Output only the writeup."
    )
    async with ctx.typing():
        try:
            lore = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"⚠️ Serverlore generation failed: {e}")
            lore = "This server's history is shrouded in mystery..."
    def build_embed(text):
        embed = discord.Embed(title=f"📜 The Chronicles of {guild.name}", description=text, color=discord.Color.dark_teal())
        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)
        return embed
    view = LanguageToggleView(prompt, build_embed)
    sent = await ctx.send(embed=build_embed(lore), view=view)
    view.message = sent

@bot.command(help="Generates a trading-card-style profile view for a user, with AI-generated art.")
async def card(ctx, member: discord.Member = None):
    member = member or ctx.author
    gid, uid = str(ctx.guild.id), str(member.id)
    msg_count = message_counts.get(gid, {}).get(uid, 0)
    reps = reputation.get(uid, 0)
    badges = get_earned_badges(member, ctx.guild)
    tiers = [(5, "🌟 LEGENDARY"), (3, "💎 EPIC"), (1, "🔷 RARE"), (0, "⚪ COMMON")]
    tier = next(label for count, label in tiers if len(badges) >= count)
    async with ctx.typing():
        art_bytes = await get_birthday_image_bytes(
            f"A stylized fantasy trading card background/portrait frame, vibrant colors, "
            f"digital art style, no readable text, themed around the concept of '{tier.split()[-1].lower()}' rarity."
        )
    embed = discord.Embed(title=f"🃏 {member.display_name}", description=f"**{tier}**", color=discord.Color.random())
    embed.add_field(name="💬 Messages", value=str(msg_count), inline=True)
    embed.add_field(name="⭐ Reputation", value=str(reps), inline=True)
    embed.add_field(name="🏅 Badges", value=str(len(badges)), inline=True)
    if badges:
        embed.add_field(name="Earned", value=" ".join(b[0] for b in badges), inline=False)
    image_file = None
    if art_bytes:
        image_file = discord.File(io.BytesIO(art_bytes), filename="card_art.png")
        embed.set_image(url="attachment://card_art.png")
    embed.set_thumbnail(url=member.display_avatar.url)
    if image_file:
        await ctx.send(embed=embed, file=image_file)
    else:
        await ctx.send(embed=embed)

@bot.command(help="Puts a user on 'trial' -- chat votes guilty/innocent, loser gets a silly temp role.")
async def court(ctx, member: discord.Member = None):
    if member is None:
        return await ctx.send("❌ Usage: `?court @user`")
    if member.bot:
        return await ctx.send("❌ Can't put a bot on trial.")
    gid, uid = str(ctx.guild.id), str(member.id)
    msg_count = message_counts.get(gid, {}).get(uid, 0)
    badges = get_earned_badges(member, ctx.guild)
    prompt = (
        f"Invent a short, funny, harmless mock 'court charge' (1-2 sentences) against a Discord user "
        f"based on this context: they've sent {msg_count} messages this week and have these badges: "
        f"{', '.join(b[1] for b in badges) if badges else 'none'}. Make it playful courtroom-drama style. "
        f"Output only the charge."
    )
    try:
        charge = await asyncio.to_thread(get_groq_text, prompt)
    except Exception:
        charge = "Being suspiciously active in this server."
    embed = discord.Embed(
        title="⚖️ COURT IS NOW IN SESSION",
        description=f"**{member.mention} stands accused of:**\n*{charge}*\n\nVote below! 👍 Innocent  •  👎 Guilty\n(60 seconds)",
        color=discord.Color.dark_red()
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    msg = await ctx.send(embed=embed)
    await msg.add_reaction("👍")
    await msg.add_reaction("👎")
    await asyncio.sleep(60)
    msg = await ctx.channel.fetch_message(msg.id)
    innocent_votes, guilty_votes = 0, 0
    for reaction in msg.reactions:
        if str(reaction.emoji) == "👍":
            innocent_votes = reaction.count - 1
        elif str(reaction.emoji) == "👎":
            guilty_votes = reaction.count - 1
    if guilty_votes > innocent_votes:
        verdict_prompt = "Write a short, dramatic, funny 'GUILTY' verdict announcement (2 sentences) for a mock Discord trial. Output only the verdict."
        try:
            verdict_text = await asyncio.to_thread(get_groq_text, verdict_prompt)
        except Exception:
            verdict_text = "The court has spoken. Guilty as charged!"
        role = await assign_temp_role(ctx.guild, member, "🔨 Convicted", 60 * 60, color=discord.Color.dark_grey(), reason="Lost a ?court trial")
        role_note = f"\n\n{member.mention} has been given the **Convicted** role for 1 hour." if role else ""
        await ctx.send(f"⚖️ **VERDICT: GUILTY** ({guilty_votes} vs {innocent_votes})\n*{verdict_text}*{role_note}")
    else:
        verdict_prompt = "Write a short, dramatic, funny 'NOT GUILTY' verdict announcement (2 sentences) for a mock Discord trial. Output only the verdict."
        try:
            verdict_text = await asyncio.to_thread(get_groq_text, verdict_prompt)
        except Exception:
            verdict_text = "The court finds insufficient evidence. Case dismissed!"
        await ctx.send(f"⚖️ **VERDICT: NOT GUILTY** ({innocent_votes} vs {guilty_votes})\n*{verdict_text}*")

@bot.command(help="Shows a user's reputation and activity aggregated across every server the bot shares with them.")
async def globalrep(ctx, member: discord.Member = None):
    member = member or ctx.author
    uid = str(member.id)
    total_messages = 0
    shared_guilds = 0
    all_badges = set()
    for guild in bot.guilds:
        g_member = guild.get_member(member.id)
        if not g_member:
            continue
        shared_guilds += 1
        total_messages += message_counts.get(str(guild.id), {}).get(uid, 0)
        for emoji, desc in get_earned_badges(g_member, guild):
            all_badges.add((emoji, desc))
    reps = reputation.get(uid, 0)
    embed = discord.Embed(
        title=f"🌐 {member.display_name}'s Global Passport",
        description=f"Aggregated across **{shared_guilds}** server{'s' if shared_guilds != 1 else ''} shared with this bot.",
        color=discord.Color.blue()
    )
    embed.add_field(name="⭐ Reputation (global)", value=str(reps), inline=True)
    embed.add_field(name="💬 Total Messages (this week, all servers)", value=str(total_messages), inline=True)
    if all_badges:
        embed.add_field(name=f"🏅 Badges Across All Servers ({len(all_badges)})", value=" ".join(b[0] for b in all_badges), inline=False)
    embed.set_thumbnail(url=member.display_avatar.url)
    await ctx.send(embed=embed)

# --------------------------------------------------------
# 🏆 SERVER AWARDS SHOW
# --------------------------------------------------------
@bot.command(name="awards", help="AI-generated server awards show based on real activity stats.")
async def awards_show(ctx):
    guild = ctx.guild
    gid = str(guild.id)
    counts = message_counts.get(gid, {})
    top_chatter_id = max(counts, key=counts.get, default=None) if counts else None
    vc_totals = defaultdict(float)
    for pair_key, seconds in voice_time_together.get(gid, {}).items():
        a, b = pair_key.split("-")
        vc_totals[a] += seconds
        vc_totals[b] += seconds
    top_vc_id = max(vc_totals, key=vc_totals.get, default=None) if vc_totals else None
    guild_marriage_stats = get_guild_marriage_stats(guild.id)
    marriage_candidates = {uid: c for uid, c in guild_marriage_stats.items() if guild.get_member(int(uid))}
    most_married_id = max(marriage_candidates, key=marriage_candidates.get, default=None) if marriage_candidates else None
    rep_candidates = {uid: r for uid, r in reputation.items() if guild.get_member(int(uid))}
    top_rep_id = max(rep_candidates, key=rep_candidates.get, default=None) if rep_candidates else None
    categories = []
    if top_chatter_id:
        m = guild.get_member(int(top_chatter_id))
        if m: categories.append(("💬 Chattiest", m, f"{counts[top_chatter_id]} messages this week"))
    if top_vc_id:
        m = guild.get_member(int(top_vc_id))
        if m: categories.append(("🎧 Voice Chat MVP", m, f"{round(vc_totals[top_vc_id]/60)} minutes in VC"))
    if most_married_id:
        m = guild.get_member(int(most_married_id))
        if m: categories.append(("💍 Serial Romantic", m, f"married {marriage_candidates[most_married_id]} times"))
    if top_rep_id:
        m = guild.get_member(int(top_rep_id))
        if m: categories.append(("⭐ Most Respected", m, f"{rep_candidates[top_rep_id]} reputation points"))
    if not categories:
        return await ctx.send("📭 Not enough server activity yet to hand out awards.")
    async with ctx.typing():
        embed = discord.Embed(title=f"🏆 The {guild.name} Awards", color=discord.Color.gold())
        for title, member, stat in categories:
            prompt = f"Write one short, funny award citation sentence (max 20 words) for winning '{title}' with this stat: {stat}. Output only the sentence."
            try:
                blurb = await asyncio.to_thread(get_groq_text, prompt)
            except Exception:
                blurb = stat
            embed.add_field(name=f"{title}: {member.display_name}", value=truncate_text(blurb.strip(), 200), inline=False)
        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)
        await ctx.send(embed=embed)

# --------------------------------------------------------
# 🎭 IMPERSONATOR
# --------------------------------------------------------
active_impersonations = {}

@bot.command(name="impersonate", help="AI mimics a random active member's writing style -- guess who it is!")
async def impersonate_game(ctx):
    async with ctx.typing():
        history = [m async for m in ctx.channel.history(limit=300) if not m.author.bot and m.content.strip()]
        by_author = defaultdict(list)
        for m in history:
            by_author[m.author.id].append(m.content)
        candidates = {uid: msgs for uid, msgs in by_author.items() if len(msgs) >= 5}
        if not candidates:
            return await ctx.send("📭 Not enough recent chat history here to impersonate anyone yet.")
        target_id = random.choice(list(candidates.keys()))
        target_member = ctx.guild.get_member(target_id)
        if not target_member:
            return await ctx.send("📭 Couldn't find a valid target, try again.")
        samples = " | ".join(candidates[target_id][-20:])[:1200]
        prompt = (
            f"Here are real messages actually sent by ONE specific Discord user (and only this user):\n{samples}\n\n"
            "Write ONE new message that could believably have come from THIS SAME PERSON. It must reflect "
            "their real writing style, tone, vocabulary, capitalization/punctuation habits, slang, and "
            "emoji use from the samples above. Where natural, reuse specific topics, words, or phrasing "
            "that actually appear in their samples rather than inventing an unrelated topic from scratch. "
            "Do not write something generic that could be from anyone -- it should feel unmistakably like "
            "it came from the same person who wrote those exact samples. Output only the message, nothing "
            "else, no quotes, no explanation."
        )
        try:
            fake_message = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"🚨 Impersonate error: {e}")
            return await ctx.send("❌ Couldn't generate an impersonation right now.")
    embed = discord.Embed(
        title="🎭 Who said this?",
        description=f"*{truncate_text(fake_message.strip(), 500)}*\n\nGuess who's being impersonated! Reply with `?guessimpersonate @user`",
        color=discord.Color.purple()
    )
    await ctx.send(embed=embed)
    active_impersonations[ctx.channel.id] = {"target_id": target_id, "guessed": False}

@bot.command(name="guessimpersonate", usage="@user", help="Guess who the bot was impersonating in the most recent ?impersonate round.")
async def guess_impersonate(ctx, member: discord.Member):
    game = active_impersonations.get(ctx.channel.id)
    if not game or game.get("guessed"):
        return await ctx.send("📭 No active impersonation round in this channel. Start one with `?impersonate`.")
    if member.id == game["target_id"]:
        game["guessed"] = True
        await ctx.send(f"🎉 Correct! {ctx.author.mention} guessed it was **{member.display_name}**!")
    else:
        await ctx.send(f"❌ Nope, not {member.display_name}. Try again!")

# --------------------------------------------------------
# 🔮 SERVER HOROSCOPE
# --------------------------------------------------------
ZODIAC_SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
horoscope_cache = {}

@bot.command(name="horoscope", usage="[sign]", help="Daily AI horoscope themed around this server. Pick a zodiac sign, or get a random one.")
async def horoscope(ctx, *, sign: str = None):
    if sign:
        sign = sign.strip().title()
        if sign not in ZODIAC_SIGNS:
            return await ctx.send(f"❌ Not a real zodiac sign. Pick from: {', '.join(ZODIAC_SIGNS)}")
    else:
        sign = random.choice(ZODIAC_SIGNS)
    today_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
    cache_key = (ctx.guild.id, sign, today_str)
    if cache_key in horoscope_cache:
        text = horoscope_cache[cache_key]
    else:
        prompt = (
            f"Write a short, funny daily horoscope (2-3 sentences) for {sign}, but themed around being a "
            f"member of a Discord server -- reference things like getting roasted, voice chat drama, "
            f"leaderboard rankings, or server gossip instead of normal horoscope topics like love/career. "
            "Output only the horoscope, nothing else."
        )
        async with ctx.typing():
            try:
                text = (await asyncio.to_thread(get_groq_text, prompt)).strip()
                horoscope_cache[cache_key] = text
            except Exception as e:
                print(f"🚨 Horoscope error: {e}")
                return await ctx.send("❌ Couldn't read the stars right now, try again in a bit.")
    embed = discord.Embed(title=f"🔮 {sign} -- Today's Server Horoscope", description=text, color=discord.Color.purple())
    await ctx.send(embed=embed)

# --------------------------------------------------------
# 🃏 AI TAROT READING
# --------------------------------------------------------
TAROT_CARDS = [
    "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
    "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
    "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
    "The Devil", "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World"
]

@bot.command(name="tarot", help="AI tarot reading, woven around your real server stats.")
async def tarot_reading(ctx, member: discord.Member = None):
    member = member or ctx.author
    gid, uid = str(ctx.guild.id), str(member.id)
    cards = random.sample(TAROT_CARDS, 3)
    badges = get_earned_badges(member, ctx.guild)
    msg_count = message_counts.get(gid, {}).get(uid, 0)
    is_married = uid in get_guild_marriages(ctx.guild.id)
    context = (
        f"Cards drawn: {', '.join(cards)}\n"
        f"Messages this week: {msg_count}\n"
        f"Badges: {', '.join(b[1] for b in badges) if badges else 'none'}\n"
        f"Married: {'yes' if is_married else 'no'}\n"
    )
    prompt = (
        f"Write a short, witty tarot reading (4-5 sentences) for a Discord user, using these three drawn "
        f"cards: {', '.join(cards)}. Weave in this real context about them naturally:\n{context}\n"
        "Make it feel like a real (but playful) tarot reading, mixing mystical tarot meaning with their "
        "actual server activity. Output only the reading, nothing else."
    )
    async with ctx.typing():
        try:
            reading = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"🚨 Tarot error: {e}")
            return await ctx.send("❌ The cards are unclear right now, try again in a bit.")
    embed = discord.Embed(
        title=f"🃏 {member.display_name}'s Tarot Reading",
        description=f"**Cards:** {', '.join(cards)}\n\n{truncate_text(reading.strip(), 1500)}",
        color=discord.Color.dark_purple()
    )
    embed.set_thumbnail(url=member.display_avatar.url)
    await ctx.send(embed=embed)

# --------------------------------------------------------
# 🔥 ROAST BATTLE
# --------------------------------------------------------
@bot.command(name="roastbattle", usage="@user1 @user2", help="AI roasts two people head-to-head based on their real activity, then declares a winner.")
async def roast_battle(ctx, user1: discord.Member, user2: discord.Member):
    if user1.id == user2.id:
        return await ctx.send("❌ Need two different people to battle.")
    async with ctx.typing():
        context1 = _birthday_context(user1, ctx.guild)
        context2 = _birthday_context(user2, ctx.guild)
        prompt = (
            f"Write a short, funny PG-13 roast battle between two Discord users. Give {user1.display_name} "
            f"one roast line based on:\n{context1}\nThen give {user2.display_name} one roast line based on:\n{context2}\n"
            f"Then declare a winner (whoever's roast material is funnier/more savage) with one closing sentence. "
            "Format as:\nROAST1: <line>\nROAST2: <line>\nWINNER: <name>\nVERDICT: <sentence>\n"
            "Keep it playful, not genuinely mean."
        )
        try:
            result = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"🚨 Roast battle error: {e}")
            return await ctx.send("❌ Couldn't start the battle right now.")
    embed = discord.Embed(
        title=f"🔥 Roast Battle: {user1.display_name} vs {user2.display_name}",
        description=truncate_text(result.strip(), 1500),
        color=discord.Color.red()
    )
    await ctx.send(embed=embed)

# --------------------------------------------------------
# 🎲 REACTION ROULETTE
# --------------------------------------------------------
async def _revert_nick_later(member, old_nick, delay):
    await asyncio.sleep(delay)
    try:
        await member.edit(nick=old_nick)
    except Exception:
        pass

async def roulette_outcome_timeout(ctx, member):
    seconds = random.choice([10, 30, 60])
    try:
        await member.timeout(discord.utils.utcnow() + timedelta(seconds=seconds), reason="?roulette")
        return f"🔇 You got hit with a {seconds}-second timeout!"
    except discord.Forbidden:
        return "🎲 Rolled a timeout, but I don't have permission to apply it. Lucky you!"

async def roulette_outcome_nickname(ctx, member):
    silly_names = ["Clown 🤡", "Chaos Goblin", "Certified Menace", "Professional Loser", "Gremlin"]
    new_nick = random.choice(silly_names)
    old_nick = member.display_name
    try:
        await member.edit(nick=new_nick[:32])
        bot.loop.create_task(_revert_nick_later(member, old_nick, 300))
        return f"📛 Your nickname is now **{new_nick}** for 5 minutes!"
    except discord.Forbidden:
        return "🎲 Rolled a nickname change, but I can't edit your nickname. Lucky you!"

async def roulette_outcome_role(ctx, member):
    role = await assign_temp_role(ctx.guild, member, "🎨 Roulette Color", 3600, color=discord.Color.random(), reason="?roulette")
    if role:
        return "🎨 You won a random colored role for 1 hour!"
    return "🎲 Rolled a color role, but couldn't create it. Lucky you!"

async def roulette_outcome_nothing(ctx, member):
    return "😌 Nothing happens. You dodged it!"

ROULETTE_OUTCOMES = [roulette_outcome_timeout, roulette_outcome_nickname, roulette_outcome_role, roulette_outcome_nothing, roulette_outcome_nothing]

@bot.command(name="roulette", help="Spin for a random (harmless) consequence -- opt-in chaos.")
async def roulette(ctx):
    outcome_fn = random.choice(ROULETTE_OUTCOMES)
    async with ctx.typing():
        result = await outcome_fn(ctx, ctx.author)
    await ctx.send(f"🎲 {ctx.author.mention} spun the wheel...\n{result}")

# --------------------------------------------------------
# 💘 FAKE DATING SHOW
# --------------------------------------------------------
@bot.command(name="datingshow", help="AI picks 3 random unmarried members and generates a silly 'who would you pick' scenario.")
async def dating_show(ctx):
    guild_marriages = get_guild_marriages(ctx.guild.id)
    eligible = [m for m in ctx.guild.members if not m.bot and str(m.id) not in guild_marriages]
    if len(eligible) < 3:
        return await ctx.send("📭 Not enough unmarried members here for a dating show yet.")
    contestants = random.sample(eligible, 3)
    context = "\n".join(f"- {m.display_name}: {_birthday_context(m, ctx.guild)}" for m in contestants)
    prompt = (
        f"Write a short, funny 'reality dating show' scenario (4-6 sentences) introducing these three "
        f"Discord members as contestants, based on real context about them:\n{context}\n"
        "Make it playful and over-the-top like a dating show narrator introduction. Output only the scenario."
    )
    async with ctx.typing():
        try:
            scenario = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"🚨 Dating show error: {e}")
            return await ctx.send("❌ Couldn't cast the show right now, try again in a bit.")
    embed = discord.Embed(title="💘 Tonight On... The Dating Show", description=truncate_text(scenario.strip(), 1500), color=discord.Color.magenta())
    embed.add_field(name="Contestants", value=", ".join(m.mention for m in contestants), inline=False)
    await ctx.send(embed=embed)

# --------------------------------------------------------
# 🖼️ CURSED COMBO GENERATOR
# --------------------------------------------------------
CURSED_SCENARIOS = [
    "running a lemonade stand during a zombie apocalypse", "being trapped in an elevator for 8 hours",
    "co-hosting a cooking show that only makes cereal", "starting a band that only plays kazoo covers",
    "opening a haunted laundromat", "being forced to compete on a reality TV show about competitive napping",
    "running for mayor of a town that doesn't exist", "being the two finalists in a hot dog eating contest neither of them wanted to enter",
]

@bot.command(name="cursedcombo", usage="[@user1] [@user2]", help="Mashes two random (or mentioned) members + a random scenario into an absurd AI short story.")
async def cursed_combo(ctx, user1: discord.Member = None, user2: discord.Member = None):
    members_pool = [m for m in ctx.guild.members if not m.bot]
    if len(members_pool) < 2:
        return await ctx.send("📭 Not enough members here for a cursed combo.")
    if not user1 or not user2:
        picks = random.sample(members_pool, 2)
        user1 = user1 or picks[0]
        user2 = user2 or (picks[1] if picks[0].id != user1.id else picks[0])
        if user1.id == user2.id:
            user2 = next(m for m in members_pool if m.id != user1.id)
    scenario = random.choice(CURSED_SCENARIOS)
    prompt = (
        f"Write a short, absurd, funny story (4-6 sentences) starring two Discord users named "
        f"{user1.display_name} and {user2.display_name}, who find themselves {scenario}. "
        "Make it chaotic and ridiculous. Output only the story."
    )
    async with ctx.typing():
        try:
            story = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"🚨 Cursed combo error: {e}")
            return await ctx.send("❌ Couldn't generate a cursed combo right now.")
    embed = discord.Embed(
        title=f"🖼️ Cursed Combo: {user1.display_name} + {user2.display_name}",
        description=f"*{scenario}*\n\n{truncate_text(story.strip(), 1500)}",
        color=discord.Color.dark_orange()
    )
    await ctx.send(embed=embed)

# --------------------------------------------------------
# 🤔 WOULD YOU RATHER
# --------------------------------------------------------
async def build_wyr_embed(author: discord.abc.User) -> Optional[discord.Embed]:
    prompt = (
        "Generate one 'Would You Rather' question for a Discord server game. Mix up the tone each time -- "
        "it can be funny, absurd, flirty, deep, wholesome, or a little sad/thought-provoking, not just "
        "silly every time. Reply in EXACTLY this format, nothing else:\n"
        "A: <first option>\nB: <second option>\n"
        "Keep each option to one short, punchy sentence -- no numbering, no extra commentary, "
        "make the two options genuinely hard to choose between."
    )
    try:
        result = await asyncio.to_thread(get_groq_text, prompt)
    except Exception as e:
        print(f"🚨 WYR error: {e}")
        return None

    a_match = re.search(r"A:\s*(.+)", result)
    b_match = re.search(r"B:\s*(.+)", result)
    option_a = a_match.group(1).strip() if a_match else None
    option_b = b_match.group(1).strip() if b_match else None
    if not option_a or not option_b:
        lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            option_a, option_b = lines[0], lines[1]
        else:
            return None

    embed = discord.Embed(
        title="🤔 Would You Rather...",
        description=f"🅰️ **{truncate_text(option_a, 200)}**\n\n**...OR...**\n\n🅱️ **{truncate_text(option_b, 200)}**",
        color=discord.Color.blurple()
    )
    embed.set_footer(text=f"Asked by {author.display_name} -- react to vote, or hit Next Question for another!")
    return embed

class WYRView(ui.View):
    """Not restricted to the original asker -- anyone can keep the round going with Next Question."""
    def __init__(self):
        super().__init__(timeout=600)
        self.message = None

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass

    @ui.button(label="Next Question ▶️", style=discord.ButtonStyle.primary)
    async def next_question(self, interaction: Interaction, button: ui.Button):
        await interaction.response.defer()
        embed = await build_wyr_embed(interaction.user)
        if embed is None:
            return await interaction.followup.send("❌ Couldn't come up with one right now, try again in a bit.", ephemeral=True)
        try:
            await interaction.message.edit(embed=embed, view=self)
            await interaction.message.clear_reactions()
            await interaction.message.add_reaction("🅰️")
            await interaction.message.add_reaction("🅱️")
        except Exception as e:
            print(f"⚠️ Failed to advance ?wyr: {e}")

@bot.command(name="wyr", help="AI-generated 'Would You Rather' -- react 🅰️ or 🅱️ to vote, hit Next Question to keep going.")
async def would_you_rather(ctx):
    embed = await build_wyr_embed(ctx.author)
    if embed is None:
        return await ctx.send("❌ Couldn't come up with one right now, try again in a bit.")
    view = WYRView()
    msg = await ctx.send(embed=embed, view=view)
    view.message = msg
    try:
        await msg.add_reaction("🅰️")
        await msg.add_reaction("🅱️")
    except discord.Forbidden:
        print("⚠️ Missing permission to add reactions for ?wyr.")

# --------------------------------------------------------
# 🎯 TRUTH OR DARE
# --------------------------------------------------------
class TruthOrDareNextView(ui.View):
    """Posted alongside the truth/dare result -- lets the same person keep playing without retyping the command."""
    def __init__(self, target: discord.Member):
        super().__init__(timeout=120)
        self.target = target
        self.message = None

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass

    @ui.button(label="Next ▶️", style=discord.ButtonStyle.success)
    async def next_round(self, interaction: Interaction, button: ui.Button):
        if interaction.user.id != self.target.id:
            return await interaction.response.send_message("Only the last player can start the next round -- run `?truthordare` yourself to play!", ephemeral=True)
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(view=self)
        embed = discord.Embed(title="🎯 Truth or Dare!", description=f"{self.target.mention}, choose your fate...", color=discord.Color.gold())
        view = TruthOrDareView(self.target)
        sent = await interaction.channel.send(embed=embed, view=view)
        view.message = sent

class TruthOrDareView(ui.View):
    def __init__(self, target: discord.Member):
        super().__init__(timeout=60)
        self.target = target
        self.message = None

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if self.message:
            try:
                await self.message.edit(content=f"⌛ {self.target.mention} didn't choose in time.", view=self)
            except Exception:
                pass

    async def _check(self, interaction: Interaction) -> bool:
        if interaction.user.id != self.target.id:
            await interaction.response.send_message("This isn't your turn to choose!", ephemeral=True)
            return False
        return True

    @ui.button(label="Truth", style=discord.ButtonStyle.primary, emoji="🗣️")
    async def truth(self, interaction: Interaction, button: ui.Button):
        if not await self._check(interaction):
            return
        await interaction.response.defer()
        prompt = (
            "Generate one 'truth' question for a game of Truth or Dare on a Discord server. Mix up the "
            "tone each time -- it can be funny, flirty, deep/emotional, a little embarrassing, or "
            "wholesome, not just silly every time. Avoid anything genuinely explicit, hateful, or that "
            "would out someone's real private/identifying information. Output only the question, nothing else."
        )
        try:
            text = await asyncio.to_thread(get_groq_text, prompt)
        except Exception:
            text = "What's the most embarrassing thing that's happened to you in the last year?"
        for child in self.children:
            child.disabled = True
        try:
            await interaction.message.edit(view=self)
        except Exception:
            pass
        embed = discord.Embed(title=f"🗣️ Truth for {self.target.display_name}", description=truncate_text(text.strip(), 500), color=discord.Color.blue())
        next_view = TruthOrDareNextView(self.target)
        sent = await interaction.followup.send(embed=embed, view=next_view)
        next_view.message = sent

    @ui.button(label="Dare", style=discord.ButtonStyle.danger, emoji="🔥")
    async def dare(self, interaction: Interaction, button: ui.Button):
        if not await self._check(interaction):
            return
        await interaction.response.defer()
        prompt = (
            "Generate one 'dare' challenge for a game of Truth or Dare on a Discord server -- text-based "
            "things only (e.g. typing in all caps for a few messages, sending an emoji-only story, using "
            "a silly nickname for 10 minutes, sending a compliment to someone in the chat, confessing a "
            "small secret). Mix up the tone -- funny, flirty, or a little bold is fine, but nothing "
            "offline, unsafe, or genuinely explicit. Output only the dare, nothing else."
        )
        try:
            text = await asyncio.to_thread(get_groq_text, prompt)
        except Exception:
            text = "Type your next 3 messages using only emojis."
        for child in self.children:
            child.disabled = True
        try:
            await interaction.message.edit(view=self)
        except Exception:
            pass
        embed = discord.Embed(title=f"🔥 Dare for {self.target.display_name}", description=truncate_text(text.strip(), 500), color=discord.Color.red())
        next_view = TruthOrDareNextView(self.target)
        sent = await interaction.followup.send(embed=embed, view=next_view)
        next_view.message = sent

@bot.command(name="truthordare", aliases=["tod"], usage="[@user]", help="Truth or Dare -- pick a target (defaults to yourself), they choose Truth or Dare, and the AI generates one. Keep going with the Next button.")
async def truth_or_dare(ctx, member: discord.Member = None):
    target = member or ctx.author
    if target.bot:
        return await ctx.send("❌ Can't play Truth or Dare with a bot.")
    embed = discord.Embed(
        title="🎯 Truth or Dare!",
        description=f"{target.mention}, choose your fate...",
        color=discord.Color.gold()
    )
    view = TruthOrDareView(target)
    view.message = await ctx.send(embed=embed, view=view)

# --------------------------------------------------------
# 🆚 WHO WOULD YOU (picks two random members from THIS server)
# --------------------------------------------------------
# Straight choose-between-two-people -- no scenario/question, just vote for
# whoever you'd pick. Whoever gets FEWER votes gets a cursed temporary role
# for 24 hours (auto-removed and auto-deleted -- reuses the same
# assign_temp_role/cleanup machinery as ?court and ?roulette).
WWY_LOSER_ROLE_NAMES = [
    "🚫 Nobody Wants You", "😢 Town Reject", "🥲 Least Chosen", "🤡 The Unwanted One",
    "💔 Voted Off", "🐢 Last Pick", "👻 Invisible Friend",
]
WWY_VOTE_SECONDS = 60

def get_wwy_candidate_pool(guild: discord.Guild) -> list[discord.Member]:
    # BUG FIX: only ever pull from THIS guild's own member list (guild.members
    # is always scoped to that specific Guild object -- it can't contain
    # another server's members). If the pool ever looked wrong before, the
    # most likely cause was an incomplete member cache on a large server; the
    # ?whowouldyou command below forces a fresh guild.chunk() first to avoid that.
    return [m for m in guild.members if not m.bot]

class WhoWouldYouNextView(ui.View):
    def __init__(self, guild: discord.Guild):
        super().__init__(timeout=180)
        self.guild = guild
        self.message = None

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass

    @ui.button(label="Next Round ▶️", style=discord.ButtonStyle.primary)
    async def next_round(self, interaction: Interaction, button: ui.Button):
        await interaction.response.defer()
        for child in self.children:
            child.disabled = True
        try:
            await interaction.message.edit(view=self)
        except Exception:
            pass
        await run_whowouldyou_round(interaction.channel, self.guild)

async def run_whowouldyou_round(channel: discord.abc.Messageable, guild: discord.Guild):
    if not guild.chunked:
        try:
            await guild.chunk()
        except Exception as e:
            print(f"⚠️ Guild chunk failed for ?whowouldyou: {e}")

    members_pool = get_wwy_candidate_pool(guild)
    if len(members_pool) < 2:
        await channel.send("📭 Not enough members here to play this yet.")
        return

    user_a, user_b = random.sample(members_pool, 2)
    embed = discord.Embed(
        title="🆚 Who Would You Choose?",
        description=f"🅰️ {user_a.mention}\n\n**VS**\n\n🅱️ {user_b.mention}",
        color=discord.Color.orange()
    )
    embed.set_footer(text=f"React to vote -- results in {WWY_VOTE_SECONDS}s. Whoever gets FEWER votes gets a cursed 24h role!")
    msg = await channel.send(embed=embed)
    try:
        await msg.add_reaction("🅰️")
        await msg.add_reaction("🅱️")
    except discord.Forbidden:
        print("⚠️ Missing permission to add reactions for ?whowouldyou.")

    await asyncio.sleep(WWY_VOTE_SECONDS)

    try:
        msg = await channel.fetch_message(msg.id)
    except Exception:
        return

    votes_a = votes_b = 0
    for reaction in msg.reactions:
        if str(reaction.emoji) == "🅰️":
            votes_a = max(0, reaction.count - 1)
        elif str(reaction.emoji) == "🅱️":
            votes_b = max(0, reaction.count - 1)

    result_lines = [f"🅰️ {user_a.mention}: **{votes_a}** votes", f"🅱️ {user_b.mention}: **{votes_b}** votes"]

    if votes_a == votes_b:
        result_lines.append("\n🤝 It's a tie -- no cursed role this round!")
    else:
        loser = user_b if votes_a > votes_b else user_a
        role_name = random.choice(WWY_LOSER_ROLE_NAMES)
        role = await assign_temp_role(
            guild, loser, role_name, 60 * 60 * 24,
            color=discord.Color.dark_grey(), reason="Lost a ?whowouldyou round"
        )
        if role:
            result_lines.append(f"\n💀 {loser.mention} got fewer votes and is now cursed with **{role_name}** for 24 hours!")
        else:
            result_lines.append(f"\n⚠️ {loser.mention} got fewer votes, but I couldn't create/assign the role (check my **Manage Roles** permission and role position).")

    result_embed = discord.Embed(title="📊 Results!", description="\n".join(result_lines), color=discord.Color.gold())
    view = WhoWouldYouNextView(guild)
    sent = await channel.send(embed=result_embed, view=view)
    view.message = sent

@bot.command(name="whowouldyou", aliases=["wwy"], help="Picks two random real members of THIS server -- react 🅰️ or 🅱️ to vote. Whoever gets fewer votes gets a cursed role for 24 hours.")
async def who_would_you(ctx):
    await run_whowouldyou_round(ctx.channel, ctx.guild)

# --------------------------------------------------------
# 📸 CAPTION BATTLE
# --------------------------------------------------------
# --------------------------------------------------------
active_caption_battles = {}
CAPTION_BATTLE_IMAGE_PROMPTS = [
    "a raccoon wearing a business suit giving a presentation, digital art",
    "a cat astronaut floating in space holding a slice of pizza, digital art",
    "a confused looking golden retriever wearing a crown sitting on a throne, digital art",
    "a penguin DJ at a nightclub, neon lights, digital art",
    "a dinosaur riding a skateboard through a shopping mall, digital art",
]

@bot.command(name="captionbattle", help="Bot posts a random weird image -- submit a caption with ?submitcaption <text>, then chat votes on the winner.")
async def caption_battle(ctx):
    if ctx.channel.id in active_caption_battles:
        return await ctx.send("❌ There's already an active caption battle in this channel. Finish it first.")
    prompt = random.choice(CAPTION_BATTLE_IMAGE_PROMPTS)
    async with ctx.typing():
        image_bytes = await generate_pollinations_image(prompt)
    if not image_bytes:
        return await ctx.send("❌ Couldn't generate an image right now, try again in a bit.")
    file = discord.File(io.BytesIO(image_bytes), filename="battle.png")
    embed = discord.Embed(title="📸 Caption Battle!", description="Submit your caption with `?submitcaption <your caption>`. Voting starts in 60 seconds!", color=discord.Color.blue())
    embed.set_image(url="attachment://battle.png")
    await ctx.send(embed=embed, file=file)
    active_caption_battles[ctx.channel.id] = {"captions": {}}
    await asyncio.sleep(60)
    battle = active_caption_battles.get(ctx.channel.id)
    if not battle or len(battle["captions"]) < 2:
        active_caption_battles.pop(ctx.channel.id, None)
        return await ctx.send("📭 Not enough captions submitted (need at least 2) -- caption battle cancelled.")
    entries = list(battle["captions"].items())[:10]
    description = "\n".join(f"{NUMBER_EMOJIS[i]} <@{uid}>: {cap}" for i, (uid, cap) in enumerate(entries))
    vote_embed = discord.Embed(title="🗳️ Vote for the best caption!", description=description, color=discord.Color.green())
    vote_msg = await ctx.send(embed=vote_embed)
    for i in range(len(entries)):
        try:
            await vote_msg.add_reaction(NUMBER_EMOJIS[i])
        except Exception:
            break
    active_caption_battles.pop(ctx.channel.id, None)

@bot.command(name="submitcaption", usage="<caption>", help="Submit your caption for the active ?captionbattle in this channel.")
async def submit_caption(ctx, *, caption: str):
    battle = active_caption_battles.get(ctx.channel.id)
    if not battle:
        return await ctx.send("❌ No active caption battle in this channel. Start one with `?captionbattle`.")
    battle["captions"][ctx.author.id] = truncate_text(caption, 150)
    try:
        await ctx.message.add_reaction("✅")
    except Exception:
        pass


# --------------------------------------------------------
# ⚖️ AI MEDIATOR
# --------------------------------------------------------
@bot.command(name="mediate", usage="@user1 @user2", help="AI reads the recent back-and-forth between two people and gives a neutral summary + lighthearted verdict.")
async def mediate(ctx, user1: discord.Member, user2: discord.Member):
    if user1.id == user2.id:
        return await ctx.send("❌ Need two different people to mediate between.")
    async with ctx.typing():
        relevant = []
        async for m in ctx.channel.history(limit=200):
            if m.author.id in (user1.id, user2.id) and m.content.strip():
                relevant.append(m)
        relevant.reverse()
        if len(relevant) < 4:
            return await ctx.send("📭 Not enough recent messages between them in this channel to mediate anything.")
        transcript = "\n".join(f"{m.author.display_name}: {m.content}" for m in relevant[-60:])
        prompt = (
            f"Here is a recent exchange between two Discord users, {user1.display_name} and {user2.display_name}:\n\n"
            f"{transcript}\n\n"
            "Act as a neutral, slightly humorous mediator. Give:\n"
            "1. A short, fair summary of each side's point (1-2 sentences each)\n"
            "2. A lighthearted 'verdict' on who has more of a point, or that it's a wash\n"
            "Stay genuinely neutral and don't escalate either side. Keep the whole reply under 150 words."
        )
        try:
            response = await asyncio.to_thread(get_groq_text, prompt)
        except Exception as e:
            print(f"🚨 Mediate error: {e}")
            return await ctx.send("❌ Couldn't mediate right now, try again in a bit.")
        embed = discord.Embed(title=f"⚖️ Mediation: {user1.display_name} vs {user2.display_name}", description=response, color=discord.Color.blue())
        await ctx.send(embed=embed)

# --------------------------------------------------------
# 🧬 CLOSEST CONNECTION FINDER
# --------------------------------------------------------
@bot.command(name="twin", usage="[@user]", help="Finds who you (or the mentioned user) interact with the most -- voice time together, replies, and being active in chat at the same time.")
async def twin(ctx, member: discord.Member = None):
    target = member or ctx.author
    gid = str(ctx.guild.id)
    async with ctx.typing():
        vc_seconds = defaultdict(float)
        for pair_key, seconds in voice_time_together.get(gid, {}).items():
            a_str, b_str = pair_key.split("-")
            if a_str == str(target.id):
                vc_seconds[int(b_str)] += seconds
            elif b_str == str(target.id):
                vc_seconds[int(a_str)] += seconds
        reply_counts = defaultdict(int)
        coactivity_counts = defaultdict(int)
        history = [m async for m in ctx.channel.history(limit=300) if not m.author.bot]
        history.reverse()
        for i, msg in enumerate(history):
            ref_author = None
            if msg.reference:
                if msg.reference.resolved and isinstance(msg.reference.resolved, discord.Message):
                    ref_author = msg.reference.resolved.author
                elif msg.reference.message_id:
                    try:
                        ref_msg = await ctx.channel.fetch_message(msg.reference.message_id)
                        ref_author = ref_msg.author
                    except Exception:
                        ref_author = None
            if ref_author and not ref_author.bot:
                if msg.author.id == target.id and ref_author.id != target.id:
                    reply_counts[ref_author.id] += 1
                elif ref_author.id == target.id and msg.author.id != target.id:
                    reply_counts[msg.author.id] += 1
            if i > 0:
                prev = history[i - 1]
                gap = (msg.created_at - prev.created_at).total_seconds()
                if 0 < gap <= 180 and prev.author.id != msg.author.id:
                    if msg.author.id == target.id:
                        coactivity_counts[prev.author.id] += 1
                    elif prev.author.id == target.id:
                        coactivity_counts[msg.author.id] += 1
        candidate_ids = (set(vc_seconds) | set(reply_counts) | set(coactivity_counts)) - {target.id}
        if not candidate_ids:
            return await ctx.send(f"📭 Not enough voice time or chat activity with anyone yet to find a match for {target.display_name}.")
        def combined_score(uid):
            return (vc_seconds.get(uid, 0) / 60.0) * 3 + reply_counts.get(uid, 0) * 2 + coactivity_counts.get(uid, 0) * 1
        best_uid = max(candidate_ids, key=combined_score)
        best_member = ctx.guild.get_member(best_uid)
        if not best_member:
            return await ctx.send("📭 Found a match, but they're no longer in this server.")
        vc_minutes = round(vc_seconds.get(best_uid, 0) / 60)
        replies = reply_counts.get(best_uid, 0)
        coactivity = coactivity_counts.get(best_uid, 0)
        breakdown = []
        if vc_minutes > 0:
            breakdown.append(f"🎧 **{vc_minutes}** minutes together in voice")
        if replies > 0:
            breakdown.append(f"↩️ **{replies}** replies exchanged")
        if coactivity > 0:
            breakdown.append(f"🕐 **{coactivity}** times active back-to-back in chat")
        embed = discord.Embed(title=f"🧬 {target.display_name}'s Closest Connection", description=f"**{best_member.mention}**\n\n" + "\n".join(breakdown), color=discord.Color.teal())
        embed.set_thumbnail(url=best_member.display_avatar.url)
        embed.set_footer(text="Based on VC time together, replies, and simultaneous chat activity")
        await ctx.send(embed=embed)

# --------------------------------------------------------
# 🗳️ ANONYMOUS CONFESSIONS (AI-screened)
# --------------------------------------------------------
async def classify_confession(text: str) -> tuple[str, str]:
    prompt = (
        "You are a content safety classifier for an anonymous confession box on a Discord server. "
        f"Classify this submitted confession:\n\n\"{text}\"\n\n"
        "Reply in EXACTLY this format, nothing else:\n"
        "LABEL: SAFE\n"
        "or\n"
        "LABEL: BLOCK\nREASON: <short reason>\n"
        "or\n"
        "LABEL: ESCALATE\nREASON: <short reason>\n\n"
        "Use BLOCK for hate speech, harassment, doxxing/identifying info about a specific real person in a "
        "harmful way, illegal content, or spam. Use ESCALATE if the confession expresses suicidal ideation, "
        "self-harm intent, or a credible threat of violence against a real person -- these need a human to "
        "see them, not to be posted publicly. Use SAFE for everything else, including embarrassing, funny, "
        "or emotionally vulnerable but non-dangerous confessions."
    )
    try:
        result = await asyncio.to_thread(get_groq_text, prompt)
    except Exception as e:
        print(f"⚠️ Confession classification failed, defaulting to BLOCK: {e}")
        return "BLOCK", "Safety check is temporarily unavailable, try again shortly."
    label_match = re.search(r"LABEL:\s*(SAFE|BLOCK|ESCALATE)", result, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", result)
    label = label_match.group(1).upper() if label_match else "BLOCK"
    reason = reason_match.group(1).strip() if reason_match else "Didn't pass the safety check."
    return label, reason

@bot.command(name="confess", usage="<message>", help="Posts your message anonymously to the confessions channel. Works in a server channel, or DM the bot directly. AI-screened for safety first.")
async def confess(ctx, *, message: str):
    target_guild = None
    if ctx.guild is not None:
        try:
            await ctx.message.delete()
        except Exception:
            pass
        target_guild = ctx.guild
    else:
        candidate_guilds = [g for g in bot.guilds if g.get_member(ctx.author.id) and get_configured_channel(g, "confessions_channel_id")]
        if not candidate_guilds:
            return await ctx.author.send("❌ None of your mutual servers with me have a confessions channel set up yet.")
        if len(candidate_guilds) == 1:
            target_guild = candidate_guilds[0]
        else:
            parts = message.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                matched = discord.utils.get(candidate_guilds, id=int(parts[0]))
                if matched:
                    target_guild = matched
                    message = parts[1]
            if target_guild is None:
                names = "\n".join(f"- **{g.name}** -- ID `{g.id}`" for g in candidate_guilds)
                return await ctx.author.send(f"❌ You're in multiple servers with a confessions channel set up. Prefix your message with the server ID: `?confess <server_id> <message>`\n{names}")

    confessions_channel = get_configured_channel(target_guild, "confessions_channel_id")
    if not confessions_channel:
        try:
            await ctx.author.send("❌ This server hasn't set up a confessions channel yet. A mod needs to run `?setchannel confessions #channel`.")
        except Exception:
            pass
        return

    label, reason = await classify_confession(message)

    if label == "SAFE":
        gid = str(target_guild.id)
        confessions_data[gid] = confessions_data.get(gid, 0) + 1
        save_data(confessions_data, CONFESSIONS_FILE)
        embed = discord.Embed(title=f"🗳️ Anonymous Confession #{confessions_data[gid]}", description=message, color=discord.Color.dark_purple(), timestamp=datetime.now(timezone.utc))
        try:
            await confessions_channel.send(embed=embed)
        except Exception as e:
            print(f"⚠️ Failed to post confession: {e}")
            return
        try:
            await ctx.author.send(f"✅ Your confession was posted anonymously in **{target_guild.name}**.")
        except Exception:
            pass
    elif label == "ESCALATE":
        try:
            await ctx.author.send(
                "I didn't post that one -- it read like you might be going through something serious, "
                "and I'd rather make sure you're okay than post it anonymously.\n\n"
                "If you're in crisis, please reach out: **988** (US Suicide & Crisis Lifeline -- call or text), "
                "or text HOME to **741741** (Crisis Text Line). Outside the US, https://findahelpline.com can "
                "point you to a local line.\n\n"
                "A mod here may also reach out to check in with you."
            )
        except Exception:
            pass
        mod_log = get_mod_log_channel(target_guild)
        if mod_log:
            try:
                await mod_log.send(f"🚨 **Confession safety escalation** from {ctx.author.mention} ({ctx.author.id}) -- not posted publicly. Reason: {reason}\n> {message}")
            except Exception as e:
                print(f"⚠️ Failed to send confession escalation alert: {e}")
    else:
        try:
            await ctx.author.send(f"❌ Your confession wasn't posted: {reason}")
        except Exception:
            pass

# --------------------------------------------------------
# 🧬 FAMILY TREE DIAGRAM
# --------------------------------------------------------
def build_family_tree_image(guild: discord.Guild, center_id: Optional[int] = None) -> Optional[bytes]:
    if not FAMILY_TREE_AVAILABLE:
        return None
    G = nx.Graph()
    relevant_ids = set()
    guild_marriages = get_guild_marriages(guild.id)
    guild_friends = get_guild_friends(guild.id)
    if center_id is not None:
        relevant_ids.add(center_id)
        partner = guild_marriages.get(str(center_id))
        if partner:
            relevant_ids.add(int(partner))
        for fid in guild_friends.get(str(center_id), []):
            relevant_ids.add(int(fid))
    else:
        for uid_str in list(guild_marriages.keys()) + list(guild_friends.keys()):
            relevant_ids.add(int(uid_str))
        relevant_ids = set(list(relevant_ids)[:40])
    if not relevant_ids:
        return None
    names = {}
    for uid in relevant_ids:
        member = guild.get_member(uid)
        raw_name = member.display_name if member else f"User {uid}"
        clean_name = raw_name.encode("latin-1", errors="ignore").decode("latin-1").strip()
        names[uid] = clean_name if clean_name else f"User {uid}"
        G.add_node(uid)
    seen_marriage_pairs = set()
    for uid in relevant_ids:
        partner_str = guild_marriages.get(str(uid))
        if partner_str:
            partner_id = int(partner_str)
            pair = tuple(sorted([uid, partner_id]))
            if partner_id in relevant_ids and pair not in seen_marriage_pairs:
                seen_marriage_pairs.add(pair)
                G.add_edge(uid, partner_id, kind="marriage")
    seen_friend_pairs = set()
    for uid in relevant_ids:
        for fid_str in guild_friends.get(str(uid), []):
            fid = int(fid_str)
            pair = tuple(sorted([uid, fid]))
            if fid in relevant_ids and pair not in seen_friend_pairs and not G.has_edge(uid, fid):
                seen_friend_pairs.add(pair)
                G.add_edge(uid, fid, kind="friend")
    if G.number_of_edges() == 0:
        return None
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.8)
    marriage_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "marriage"]
    friend_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "friend"]
    nx.draw_networkx_nodes(G, pos, node_color="#5865F2", node_size=1400)
    nx.draw_networkx_labels(G, pos, labels=names, font_size=8, font_color="white")
    nx.draw_networkx_edges(G, pos, edgelist=marriage_edges, edge_color="#ED4245", width=2.5)
    nx.draw_networkx_edges(G, pos, edgelist=friend_edges, edge_color="#57F287", width=1.5, style="dashed")
    plt.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#2b2d31", dpi=150)
    plt.close()
    buf.seek(0)
    return buf.read()

@bot.command(name="familytree", usage="[@user]", help="Visual diagram of marriages (red) and friendships (green) -- server-wide, or just one person's connections.")
async def family_tree(ctx, member: discord.Member = None):
    if not FAMILY_TREE_AVAILABLE:
        return await ctx.send("❌ This feature needs `matplotlib` and `networkx` added to the bot's `requirements.txt` -- ask whoever hosts the bot to add them and redeploy.")
    async with ctx.typing():
        image_bytes = await asyncio.to_thread(build_family_tree_image, ctx.guild, member.id if member else None)
    if not image_bytes:
        msg = f"📭 {member.display_name} isn't married or listed as friends with anyone yet." if member else "📭 Not enough marriages/friendships on record yet to draw a tree."
        return await ctx.send(msg)
    file = discord.File(io.BytesIO(image_bytes), filename="family_tree.png")
    embed = discord.Embed(title=f"🧬 Family Tree{f' — {member.display_name}' if member else ''}", color=discord.Color.blurple())
    embed.set_image(url="attachment://family_tree.png")
    embed.set_footer(text="🔴 Red = married   🟢 Green (dashed) = friends")
    await ctx.send(embed=embed, file=file)

# --------------------------------------------------------
# 🗞️ WEEKLY DIGEST OPT-OUT
# --------------------------------------------------------
@bot.command(name="digestoptout", help="Opt out of the weekly personalized recap DM.")
async def digest_optout_cmd(ctx):
    uid = ctx.author.id
    if uid not in digest_optout:
        digest_optout.append(uid)
        save_data(digest_optout, DIGEST_OPTOUT_FILE)
    await ctx.send("✅ You won't get the weekly digest DM anymore. Use `?digestoptin` to turn it back on.")

@bot.command(name="digestoptin", help="Opt back in to the weekly personalized recap DM.")
async def digest_optin_cmd(ctx):
    uid = ctx.author.id
    if uid in digest_optout:
        digest_optout.remove(uid)
        save_data(digest_optout, DIGEST_OPTOUT_FILE)
    await ctx.send("✅ You'll get the weekly digest DM again.")


# --------------------------------------------------------
# 📖 ?bnstory -- AI ROLEPLAY STORY THREADS
# --------------------------------------------------------
# Flow: ?bnstory -> pick Public/Private -> thread gets created -> pick
# genre(s) -> pick a duration (e.g. 45m or 2d, min 30m) -> the AI writes
# an opening scene and roleplays every other character; the user plays
# themself by just typing in the thread. As the clock runs down the AI
# is nudged to wrap the plot up, and when time's up it writes a real
# ending, then asks if you want to go again (same thread, fresh story)
# or be done (thread auto-deletes).
#
# Session state lives in `story_sessions` (persisted to STORY_SESSIONS_FILE)
# keyed by thread ID, so an active story survives a Railway restart/redeploy.
# All the interactive Views below are registered as *persistent* views
# (custom_id + timeout=None, added via bot.add_view in on_ready) and pull
# whatever state they need out of `story_sessions` at click-time rather
# than storing it on the view instance -- that's what makes them safe to
# reuse across a restart and across many concurrent stories at once.

STORY_GENRE_META = {
    "adventure": ("🗺️", "Adventure"),
    "romantic": ("💕", "Romantic"),
    "horror": ("👻", "Horror"),
    "emotional": ("😢", "Emotional"),
    "fantasy": ("🐉", "Fantasy"),
    "fiction": ("📖", "Fiction"),
}
STORY_HISTORY_LIMIT = 40      # messages of context kept per story (both narrator + user turns)
STORY_MAX_ACTIVE_PER_USER = 2  # simple abuse guard so one person can't spin up endless threads

def save_story_sessions():
    save_data(story_sessions, STORY_SESSIONS_FILE)

def parse_story_duration(text: str) -> Optional[int]:
    """Parses '30m' / '2d' style duration strings into seconds. Returns None if invalid."""
    match = re.fullmatch(r"\s*(\d+)\s*([dm])\s*", text.strip().lower())
    if not match:
        return None
    value, unit = int(match.group(1)), match.group(2)
    if value <= 0:
        return None
    return value * 86400 if unit == "d" else value * 60

def format_story_duration(seconds: int) -> str:
    if seconds >= 86400 and seconds % 86400 == 0:
        days = seconds // 86400
        return f"{days} day{'s' if days != 1 else ''}"
    minutes = seconds // 60
    return f"{minutes} minute{'s' if minutes != 1 else ''}"

def build_story_system_prompt(genres: list[str]) -> str:
    genre_list = ", ".join(STORY_GENRE_META[g][1] for g in genres if g in STORY_GENRE_META)
    return (
        "You are the narrator and every non-player character (NPC) in an immersive, richly written text "
        f"roleplay story. Genre(s): {genre_list}. Write like it's straight out of a gripping novel -- vivid, "
        "dramatic, emotionally real, full of sensory detail and natural dialogue, never rushed or generic. "
        "The person you're roleplaying with plays the protagonist, addressed as 'you' -- NEVER speak, think, "
        "or make decisions on their behalf. Only narrate the world and voice the other characters, then stop "
        "at a natural moment for them to respond. Stay tightly consistent with everything that has already "
        "happened in the story so far. Never break character and never mention that you are an AI."
    )

async def get_story_ai_reply(session: dict, user_message: Optional[str], conclude: bool) -> str:
    now = datetime.now(timezone.utc)
    start_time = datetime.fromisoformat(session["start_time"])
    end_time = datetime.fromisoformat(session["end_time"])
    total = max(1.0, (end_time - start_time).total_seconds())
    frac_left = max(0.0, (end_time - now).total_seconds() / total)

    if conclude:
        pacing = (
            "This story must end now. Write a dramatic, emotionally satisfying FINAL scene that resolves "
            "the plot and brings the story to a clear, definitive close -- do not leave it open-ended or "
            "trail off. Around 150-300 words."
        )
    elif frac_left < 0.15:
        pacing = "Very little time is left in this story -- start moving the plot toward its climax and a resolution soon."
    elif frac_left < 0.4:
        pacing = "The story is past its midpoint -- start raising the stakes and deepening the plot."
    else:
        pacing = "The story is still early -- take your time building the scene, characters, and world."

    system_content = build_story_system_prompt(session["genres"]) + "\n\n" + pacing
    messages = [{"role": "system", "content": system_content}]
    messages.extend(session["history"][-STORY_HISTORY_LIMIT:])
    if user_message:
        messages.append({"role": "user", "content": user_message})
    elif conclude:
        messages.append({"role": "user", "content": "[The story must conclude now.]"})
    if len(messages) == 1:
        # Nothing but the system prompt (brand new story, no history yet) --
        # give it a nudge so it actually writes the opening scene.
        messages.append({"role": "user", "content": "Begin the story now."})

    def _call():
        completion = groq_client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=messages,
            max_tokens=GROQ_QWEN_MAX_TOKENS,
            **GROQ_CHAT_REASONING_KWARGS,
        )
        return completion.choices[0].message.content

    return await asyncio.to_thread(_call)

async def send_genre_prompt(thread: discord.Thread):
    embed = discord.Embed(
        title="🎭 Choose Your Genre(s)",
        description="Pick one or more genres below, then hit **Confirm**. Mixing genres blends them together (e.g. Adventure + Romantic).",
        color=discord.Color.purple()
    )
    embed.add_field(name="Options", value=" • ".join(f"{e} {n}" for e, n in STORY_GENRE_META.values()), inline=False)
    await thread.send(embed=embed, view=StoryGenreView())

async def conclude_story(thread: discord.Thread, session: dict, final_user_message: Optional[str] = None):
    if session.get("state") == "concluding":
        return  # already being wrapped up -- avoids double-firing (on_message vs. the background loop racing)
    session["state"] = "concluding"
    save_story_sessions()

    if final_user_message:
        session["history"].append({"role": "user", "content": final_user_message})

    async with thread.typing():
        try:
            ending = await get_story_ai_reply(session, user_message=None, conclude=True)
        except Exception as e:
            print(f"🚨 Story conclusion AI error in thread {thread.id}: {e}")
            ending = "*...and with that, the story quietly comes to a close.*"

    session["history"].append({"role": "assistant", "content": ending})
    session["state"] = "concluded_wait"
    save_story_sessions()

    await safe_send(thread, f"🏁 **The End**\n\n{ending}")

    embed = discord.Embed(
        title="📖 Story Finished!",
        description="Want to start another one in this thread?",
        color=discord.Color.gold()
    )
    await thread.send(embed=embed, view=StoryContinueView())

async def handle_story_message(message: discord.Message, session: dict):
    """Called from on_message for any non-command message sent inside an
    active story thread. Handles private-thread write permissions, routes
    the message to the AI, and posts its reply."""
    thread = message.channel
    author = message.author

    if session["visibility"] == "private":
        can_write = author.id == session["owner_id"] or author.id in session.get("allowed_write", [])
        if not can_write:
            can_view = author.id in session.get("allowed_view", [])
            try:
                await message.delete()
            except Exception:
                pass
            if can_view:
                try:
                    await author.send(
                        f"👀 You can view **{thread.name}** but you don't have write access yet. "
                        f"Ask the story's creator to run `?stallowwrite @{author.name}` in the thread."
                    )
                except Exception:
                    pass
            return

    now = datetime.now(timezone.utc)
    end_time = datetime.fromisoformat(session["end_time"])
    if now >= end_time:
        await conclude_story(thread, session, final_user_message=f"{author.display_name}: {message.content}")
        return

    session["history"].append({"role": "user", "content": f"{author.display_name}: {message.content}"})
    async with thread.typing():
        try:
            reply = await get_story_ai_reply(session, user_message=None, conclude=False)
        except Exception as e:
            print(f"🚨 Story AI error in thread {thread.id}: {e}")
            return await thread.send("❌ Something went wrong continuing the story -- try sending your message again.")

    session["history"] = session["history"][-STORY_HISTORY_LIMIT:]
    session["history"].append({"role": "assistant", "content": reply})
    save_story_sessions()
    await safe_send(thread, reply)

async def start_story_thread(interaction: Interaction, owner: discord.Member, visibility: str):
    channel = interaction.channel
    guild = interaction.guild
    thread_kind = discord.ChannelType.public_thread if visibility == "public" else discord.ChannelType.private_thread
    thread_name = f"{'📖' if visibility == 'public' else '🔒'} {owner.display_name}'s Story"

    try:
        thread = await channel.create_thread(name=thread_name[:100], type=thread_kind, auto_archive_duration=1440)
    except discord.Forbidden:
        return await interaction.followup.send("❌ I don't have permission to create threads here.", ephemeral=True)
    except discord.HTTPException:
        if visibility == "private":
            return await interaction.followup.send(
                "❌ Couldn't create a private thread -- this server may not support them (needs a certain "
                "boost level). Try `?bnstory` again and pick **Public** instead.", ephemeral=True
            )
        return await interaction.followup.send("❌ Couldn't create the thread -- try again in a bit.", ephemeral=True)

    try:
        await thread.add_user(owner)
    except Exception:
        pass

    story_sessions[str(thread.id)] = {
        "guild_id": guild.id,
        "owner_id": owner.id,
        "visibility": visibility,
        "state": "genre",
        "genres": [],
        "duration_seconds": None,
        "start_time": None,
        "end_time": None,
        "history": [],
        "allowed_view": [],
        "allowed_write": [],
    }
    save_story_sessions()

    await interaction.followup.send(f"📖 Story thread created: {thread.mention}", ephemeral=True)

    intro_lines = [f"📖 {owner.mention}, let's set up your story!"]
    if visibility == "private":
        intro_lines.append(
            "🔒 This is a **private** thread -- only you can see it right now. Use `?stallowsee @user` to let "
            "someone view it, or `?stallowwrite @user` to also let them join in and write."
        )
    else:
        intro_lines.append("🌍 This is a **public** thread -- anyone can join in and write.")
    await thread.send("\n".join(intro_lines))

    await send_genre_prompt(thread)


class StoryVisibilityView(ui.View):
    """Short-lived, not persistent -- picked immediately after ?bnstory, so
    surviving a mid-restart is an acceptable tradeoff for keeping this simple."""
    def __init__(self, owner: discord.Member):
        super().__init__(timeout=120)
        self.owner = owner
        self.message = None

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if self.message:
            try:
                await self.message.edit(content="⌛ Timed out -- run `?bnstory` again when you're ready.", view=self)
            except Exception:
                pass

    async def _check(self, interaction: Interaction) -> bool:
        if interaction.user.id != self.owner.id:
            await interaction.response.send_message("Only the person who ran `?bnstory` can choose this!", ephemeral=True)
            return False
        return True

    @ui.button(label="🌍 Public", style=discord.ButtonStyle.primary)
    async def public_btn(self, interaction: Interaction, button: ui.Button):
        if not await self._check(interaction):
            return
        await interaction.response.defer()
        for child in self.children:
            child.disabled = True
        await interaction.message.edit(view=self)
        await start_story_thread(interaction, self.owner, "public")

    @ui.button(label="🔒 Private", style=discord.ButtonStyle.secondary)
    async def private_btn(self, interaction: Interaction, button: ui.Button):
        if not await self._check(interaction):
            return
        await interaction.response.defer()
        for child in self.children:
            child.disabled = True
        await interaction.message.edit(view=self)
        await start_story_thread(interaction, self.owner, "private")


class StoryGenreSelect(ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label=name, emoji=emoji, value=key)
            for key, (emoji, name) in STORY_GENRE_META.items()
        ]
        super().__init__(
            custom_id="bnstory_genre_select",
            placeholder="Choose one or more genres...",
            min_values=1, max_values=len(options), options=options,
        )

    async def callback(self, interaction: Interaction):
        session = story_sessions.get(str(interaction.channel.id))
        if not session or session.get("state") != "genre":
            return await interaction.response.send_message("❌ This story setup isn't active anymore.", ephemeral=True)
        if interaction.user.id != session["owner_id"]:
            return await interaction.response.send_message("Only the story's creator can choose this!", ephemeral=True)
        session["genres"] = self.values
        save_story_sessions()
        names = ", ".join(STORY_GENRE_META[v][1] for v in self.values)
        await interaction.response.send_message(f"✅ Selected: **{names}** -- click **Confirm** when ready!", ephemeral=True)

class StoryGenreView(ui.View):
    """Persistent -- registered via bot.add_view() in on_ready. Reads
    everything it needs from story_sessions[thread_id] at click-time instead
    of storing state on the view instance, so it's safe to share one
    registered instance across every concurrent story and across restarts."""
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(StoryGenreSelect())

    @ui.button(label="Confirm ✅", style=discord.ButtonStyle.success, custom_id="bnstory_genre_confirm")
    async def confirm(self, interaction: Interaction, button: ui.Button):
        session = story_sessions.get(str(interaction.channel.id))
        if not session or session.get("state") != "genre":
            return await interaction.response.send_message("❌ This story setup isn't active anymore.", ephemeral=True)
        if interaction.user.id != session["owner_id"]:
            return await interaction.response.send_message("Only the story's creator can do that!", ephemeral=True)
        if not session.get("genres"):
            return await interaction.response.send_message("❌ Pick at least one genre from the dropdown first.", ephemeral=True)
        await interaction.response.send_modal(StoryDurationModal())


class StoryDurationModal(ui.Modal, title="Story Duration"):
    duration_input = ui.TextInput(
        label="Duration (e.g. 30m or 2d, min 30m)",
        placeholder="30m",
        max_length=10,
        required=True,
    )

    async def on_submit(self, interaction: Interaction):
        thread_id = str(interaction.channel.id)
        session = story_sessions.get(thread_id)
        if not session or session.get("state") != "genre":
            return await interaction.response.send_message("❌ This story setup isn't active anymore.", ephemeral=True)
        if interaction.user.id != session["owner_id"]:
            return await interaction.response.send_message("Only the story's creator can do that!", ephemeral=True)

        seconds = parse_story_duration(self.duration_input.value)
        if seconds is None:
            return await interaction.response.send_message(
                "❌ Couldn't parse that. Use a number followed by `d` (days) or `m` (minutes), e.g. `45m` or `2d`.",
                ephemeral=True
            )
        if seconds < 1800:
            return await interaction.response.send_message("❌ The minimum story duration is **30 minutes** (`30m`).", ephemeral=True)

        await interaction.response.defer()

        now = datetime.now(timezone.utc)
        session["duration_seconds"] = seconds
        session["start_time"] = now.isoformat()
        session["end_time"] = (now + timedelta(seconds=seconds)).isoformat()
        session["state"] = "active"
        session["history"] = []
        save_story_sessions()

        thread = interaction.channel
        genre_names = ", ".join(STORY_GENRE_META[g][1] for g in session["genres"])
        await thread.send(
            f"🎬 **Genre:** {genre_names} • **Duration:** {format_story_duration(seconds)}\n"
            f"The story begins now -- just type in this thread to play your part!"
        )

        async with thread.typing():
            try:
                opening = await get_story_ai_reply(session, user_message=None, conclude=False)
            except Exception as e:
                print(f"🚨 Story opening AI error in thread {thread.id}: {e}")
                opening = "*...the story hesitates to begin. Try saying something to kick things off.*"

        session["history"].append({"role": "assistant", "content": opening})
        save_story_sessions()
        await safe_send(thread, opening)


class StoryContinueView(ui.View):
    """Persistent -- registered via bot.add_view() in on_ready, same reasoning as StoryGenreView."""
    def __init__(self):
        super().__init__(timeout=None)

    @ui.button(label="Yes, another story!", style=discord.ButtonStyle.success, emoji="🔁", custom_id="bnstory_continue_yes")
    async def yes(self, interaction: Interaction, button: ui.Button):
        thread_id = str(interaction.channel.id)
        session = story_sessions.get(thread_id)
        if not session or session.get("state") != "concluded_wait":
            return await interaction.response.send_message("❌ This isn't waiting on a response right now.", ephemeral=True)
        if interaction.user.id != session["owner_id"]:
            return await interaction.response.send_message("Only the story's creator can choose this!", ephemeral=True)

        session["state"] = "genre"
        session["genres"] = []
        session["duration_seconds"] = None
        session["start_time"] = None
        session["end_time"] = None
        session["history"] = []
        save_story_sessions()

        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(view=self)
        await send_genre_prompt(interaction.channel)

    @ui.button(label="No, I'm done", style=discord.ButtonStyle.danger, emoji="🛑", custom_id="bnstory_continue_no")
    async def no(self, interaction: Interaction, button: ui.Button):
        thread_id = str(interaction.channel.id)
        session = story_sessions.get(thread_id)
        if not session or session.get("state") != "concluded_wait":
            return await interaction.response.send_message("❌ This isn't waiting on a response right now.", ephemeral=True)
        if interaction.user.id != session["owner_id"]:
            return await interaction.response.send_message("Only the story's creator can choose this!", ephemeral=True)

        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(content="👋 Thanks for the story! This thread will be deleted shortly.", view=self)

        story_sessions.pop(thread_id, None)
        save_story_sessions()
        bot.loop.create_task(delete_thread_later(interaction.channel, 10))


# NEW: catches stories whose timer runs out while nobody's actively typing --
# without this, a story would just sit "active" forever until someone happens
# to send another message (which is what triggers the inline expiry check in
# handle_story_message). Runs independently of that per-message check.
@tasks.loop(minutes=5.0)
async def story_expiry_check_loop():
    now = datetime.now(timezone.utc)
    for thread_id_str, session in list(story_sessions.items()):
        if session.get("state") != "active":
            continue
        try:
            end_time = datetime.fromisoformat(session["end_time"])
        except Exception:
            continue
        if now >= end_time:
            thread = bot.get_channel(int(thread_id_str))
            if not thread:
                story_sessions.pop(thread_id_str, None)
                save_story_sessions()
                continue
            bot.loop.create_task(conclude_story(thread, session))

@story_expiry_check_loop.error
async def story_expiry_check_loop_error(error):
    print(f"🚨 Story Expiry Check Loop Error: {error}")


@bot.command(name="bnstory", help="Starts an interactive AI roleplay story -- pick public/private, genre(s), and a duration, then just chat in the thread to play it out.")
async def begin_story(ctx):
    if ctx.guild is None:
        return await ctx.send("❌ `?bnstory` needs to be run in a server (threads can't be created in DMs).")

    owned = sum(
        1 for s in story_sessions.values()
        if s.get("owner_id") == ctx.author.id and s.get("guild_id") == ctx.guild.id
    )
    if owned >= STORY_MAX_ACTIVE_PER_USER:
        return await ctx.send(f"❌ You already have {STORY_MAX_ACTIVE_PER_USER} story threads going on this server. Finish or end one first.")

    embed = discord.Embed(
        title="📖 Begin a Story",
        description="Should this be a **public** thread (anyone can join and write) or a **private** thread (only you, until you invite people)?",
        color=discord.Color.blurple()
    )
    view = StoryVisibilityView(ctx.author)
    view.message = await ctx.send(embed=embed, view=view)

@bot.command(name="stallowsee", usage="@user", help="[Story creator only] Lets someone view your private story thread (run this inside the thread).")
async def story_allow_see(ctx, member: discord.Member):
    if not isinstance(ctx.channel, discord.Thread):
        return await ctx.send("❌ Run this inside the story thread.")
    session = story_sessions.get(str(ctx.channel.id))
    if not session:
        return await ctx.send("❌ This isn't an active `?bnstory` thread.")
    if session["visibility"] != "private":
        return await ctx.send("❌ This story is public -- everyone can already see it.")
    if session["owner_id"] != ctx.author.id:
        return await ctx.send("❌ Only this story's creator can do that.")

    try:
        await ctx.channel.add_user(member)
    except Exception as e:
        return await ctx.send(f"❌ Couldn't add them: {e}")

    if member.id not in session["allowed_view"]:
        session["allowed_view"].append(member.id)
    save_story_sessions()
    await ctx.send(f"👀 {member.mention} can now see this story. They're read-only for now -- use `?stallowwrite @user` to let them join in and write.")

@bot.command(name="stallowwrite", usage="@user", help="[Story creator only] Lets someone view AND write in your private story thread (run this inside the thread).")
async def story_allow_write(ctx, member: discord.Member):
    if not isinstance(ctx.channel, discord.Thread):
        return await ctx.send("❌ Run this inside the story thread.")
    session = story_sessions.get(str(ctx.channel.id))
    if not session:
        return await ctx.send("❌ This isn't an active `?bnstory` thread.")
    if session["visibility"] != "private":
        return await ctx.send("❌ This story is public -- everyone can already write here.")
    if session["owner_id"] != ctx.author.id:
        return await ctx.send("❌ Only this story's creator can do that.")

    try:
        await ctx.channel.add_user(member)
    except Exception as e:
        return await ctx.send(f"❌ Couldn't add them: {e}")

    if member.id not in session["allowed_view"]:
        session["allowed_view"].append(member.id)
    if member.id not in session["allowed_write"]:
        session["allowed_write"].append(member.id)
    save_story_sessions()
    await ctx.send(f"✍️ {member.mention} can now see and write in this story!")


# --------------------------------------------------------
# 🖼️ ?resizepfp -- CROP-FREE IMAGE RESIZER
# --------------------------------------------------------
# Takes whatever image the user attaches and pads it onto a square canvas
# (instead of stretching/cropping) so nothing in the picture gets cut off,
# then hands the result back as a file for them to download and set as
# their OWN Discord avatar. This does NOT generate or change anything about
# the image's content, and it does NOT touch the bot's own avatar -- it's a
# pure resize utility. Works the same in a server or in a DM.
def process_avatar_image(image_bytes: bytes) -> bytes:
    """Fits the image into a square without cropping OR distorting it:
    the original image is placed at its correct (untouched) aspect ratio,
    scaled to fit entirely inside the square -- and the leftover space
    around it is filled with a blurred, scaled-up copy of the SAME image
    instead of a flat black/transparent background, so it doesn't look like
    blank canvas was added. (The blurred fill layer is the only part that
    gets cropped/stretched -- it's just a decorative backdrop, never the
    actual subject.)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    side = max(w, h)

    # Blurred backdrop: scale the image up until it fully covers the square,
    # crop the overflow, then blur and dim it slightly.
    cover_scale = max(side / w, side / h)
    bg = img.resize((max(1, round(w * cover_scale)), max(1, round(h * cover_scale))), Image.LANCZOS)
    bg_w, bg_h = bg.size
    left, top = (bg_w - side) // 2, (bg_h - side) // 2
    bg = bg.crop((left, top, left + side, top + side))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=max(8, int(side * 0.04))))
    bg = Image.eval(bg, lambda px: int(px * 0.65))  # dim so the sharp foreground pops
    canvas = bg.convert("RGBA")

    # Foreground: the ORIGINAL image, full aspect ratio preserved, scaled to
    # fit entirely inside the square -- never cropped, never stretched.
    fit_scale = min(side / w, side / h)
    fg = img.resize((max(1, round(w * fit_scale)), max(1, round(h * fit_scale))), Image.LANCZOS).convert("RGBA")
    fg_w, fg_h = fg.size
    canvas.paste(fg, ((side - fg_w) // 2, (side - fg_h) // 2), fg)

    MAX_SIDE = 1024
    if side > MAX_SIDE:
        canvas = canvas.resize((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()

@bot.command(name="resizepfp", aliases=["gen"], help="Attach an image and this pads/resizes it to fit Discord's square avatar shape without cropping anything, then gives it back for you to set as your own PFP.")
async def resize_pfp(ctx):
    if not ctx.message.attachments:
        return await ctx.send("❌ Attach an image with this command, e.g. `?resizepfp` + an image attached.")

    attachment = ctx.message.attachments[0]
    if not (attachment.content_type and attachment.content_type.startswith("image/")):
        return await ctx.send("❌ That attachment isn't an image.")

    if not PFP_TOOLS_AVAILABLE:
        return await ctx.send("❌ This feature needs `Pillow` added to the bot's `requirements.txt` -- ask whoever hosts the bot to add it and redeploy.")

    raw_bytes = await attachment.read()
    try:
        processed = await asyncio.to_thread(process_avatar_image, raw_bytes)
    except Exception as e:
        return await ctx.send(f"❌ Couldn't process that image: {e}")

    file = discord.File(io.BytesIO(processed), filename="pfp_resized.png")
    await ctx.send(
        "✅ Here's your image fit to Discord's square avatar shape -- your original picture is untouched "
        "(no cropping, no stretching), and the space around it is filled with a blurred version of the "
        "same image instead of a blank background. Download it and set it as your PFP.",
        file=file
    )


class HelpView(ui.View):
    def __init__(self, bot, author):
        super().__init__(timeout=300)
        self.bot = bot
        self.author = author
        self.message = None
        self.current_page = 1

        general_pages = [
            self.get_utility_page,
            self.get_ai_page,
            self.get_fun_ai_page,
            self.get_legends_page,
            self.get_engagement_page,
            self.get_relationships_page,
            self.get_reminders_page,
            self.get_private_channels_page,
            self.get_story_page,
            self.get_new_features_page,
            self.get_games_page,
        ]
        mod_pages = [
            self.get_moderation_page,
            self.get_warnings_page,
            self.get_server_config_page,
        ]
        admin_pages = [self.get_bot_identity_page]
        owner_pages = [self.get_owner_page]

        perms = getattr(author, "guild_permissions", None)
        is_mod = bool(perms and (perms.kick_members or perms.ban_members or perms.manage_messages
                                  or perms.manage_guild or perms.administrator or perms.moderate_members))
        is_admin = bool(perms and perms.administrator)
        is_owner_username = author.name.lower() in STATUS_COMMAND_OWNERS

        self.pages = list(general_pages)
        if is_mod:
            self.pages += mod_pages
        if is_admin:
            self.pages += admin_pages
        if is_owner_username:
            self.pages += owner_pages

    def create_page(self, page_num):
        embed = discord.Embed(color=discord.Color.blue())
        embed.set_footer(text=f"Page {page_num} of {len(self.pages)} | Requested by {self.author.name}")
        return self.pages[page_num - 1](embed)

    def get_moderation_page(self, embed):
        embed.title = "🛡️ Moderation Commands"
        embed.description = "Maintain order in the server."
        embed.add_field(name="`?kick @user [reason]`", value="Kick a member.", inline=False)
        embed.add_field(name="`?ban @user_or_id [reason]`", value="Ban a member (works by mention OR raw user ID, even if they've left).", inline=False)
        embed.add_field(name="`?unban <user_id or name>`", value="Unban a user.", inline=False)
        embed.add_field(name="`?timeout @user <minutes> [reason]`", value="Timeout a member.", inline=False)
        embed.add_field(name="`?purge <amount>`", value="Bulk delete messages (max 100).", inline=False)
        embed.add_field(name="`?lock` / `?unlock`", value="Toggle @everyone's ability to send messages.", inline=False)
        embed.add_field(name="`?nick @user <new nickname>`", value="Change a member's nickname.", inline=False)
        embed.add_field(name="`?say <message>`", value="Bot repeats your message and deletes your command.", inline=False)
        embed.add_field(name="`?restrict <word>` / `?unrestrict <word>`", value="Add/remove a word from the auto-delete blacklist.", inline=False)
        embed.add_field(name="`?restrictedlist`", value="Show all currently restricted words.", inline=False)
        embed.add_field(name="`?spamem @user <emoji>`", value="Auto-reacts with that emoji on every message the user sends, until stopped.", inline=False)
        embed.add_field(name="`?stopspamem @user`", value="Stops the emoji auto-react for a user.", inline=False)
        embed.add_field(name="`?addav @user` (attach an image)", value="Sets a custom avatar override -- their `?av` shows this image everywhere this bot is, until removed.", inline=False)
        embed.add_field(name="`?removeav @user`", value="Removes a user's custom avatar override.", inline=False)
        embed.add_field(name="`?jail @user [reason]`", value="Jails a member -- they can only see/type in the configured jail channel until `?unjail`. Needs `?setjailchannel` set up first (see the Server Setup page).", inline=False)
        embed.add_field(name="`?unjail @user`", value="Releases a jailed member.", inline=False)
        embed.add_field(name="`?undo`", value="Reverses the last undoable moderation action on this server -- purge, warn, timeout, ban, kick, jail, lock/unlock, or `?resetlb`. Works for both `?commands` and the control channel.", inline=False)
        embed.add_field(name="`?resetlb @user` (`?resetleaderboard`)", value="Resets a specific user's weekly leaderboard message count to zero.", inline=False)
        return embed

    def get_warnings_page(self, embed):
        embed.title = "⚠️ Warning System"
        embed.description = "Track and manage member warnings."
        embed.add_field(name="`?warn @user [reason]`", value="Issue a warning to a member.", inline=False)
        embed.add_field(name="`?warnings @user`", value="View a member's warning history.", inline=False)
        embed.add_field(name="`?delwarn @user <id|all>`", value="Delete a specific warning or clear all of them.", inline=False)
        embed.add_field(name="`?modpoll @user <votes_needed> <kick|warn|ban|timeout> [duration_minutes if timeout] [reason]`", value="Starts a vote -- as soon as it hits the target vote count, the bot immediately carries out the action. Poll-starter's own vote doesn't count, and it expires after 5 minutes if the target isn't reached.", inline=False)
        return embed

    def get_utility_page(self, embed):
        embed.title = "🛠️ Utility & General"
        embed.description = "Useful tools and profile info."
        embed.add_field(name="`?ping`", value="Check the bot's latency.", inline=False)
        embed.add_field(name="`?gl`", value="Get a link to invite this bot to another server.", inline=False)
        embed.add_field(name="`?afk [reason]`", value="Set yourself as AFK.", inline=False)
        embed.add_field(name="`?afkclear @user`", value="Force-clear a member's AFK status (mods only).", inline=False)
        embed.add_field(name="`?av [user]`", value="Show a user's avatar (or their mod-set override, see `?addav` on the Moderation page).", inline=False)
        embed.add_field(name="`?avglobal` (`?gav`) `[user]`", value="Shows a user's GLOBAL avatar specifically, ignoring any per-server (Nitro) avatar.", inline=False)
        embed.add_field(name="`?avserver` (`?sav`) `[user]`", value="Shows a user's SERVER-SPECIFIC (Nitro) avatar for this server, if they've set one.", inline=False)
        embed.add_field(name="`?banner [user]`", value="Show a user's profile banner.", inline=False)
        embed.add_field(name="`?userinfo [user]`", value="Show join date, account age, and join/leave history.", inline=False)
        embed.add_field(name="`?hl <word>` / `?unhl <word>`", value="Add/remove a DM highlight keyword (mods only).", inline=False)
        embed.add_field(name="`?listhl`", value="List your highlight keywords (mods only).", inline=False)
        embed.add_field(name="`?steal <emoji>`", value="Clone an emoji from another server into this one.", inline=False)
        embed.add_field(name="`?schedule <movie> <MM/DD/YYYY> <HH:MM>`", value="Schedule a movie/event announcement.", inline=False)
        embed.add_field(name="`?resizepfp` (attach an image)", value="Fits an attached image into Discord's square avatar shape with no cropping and no stretching -- fills the leftover space with a blurred version of the same image instead of a blank background. Works in DM too.", inline=False)
        return embed

    def get_ai_page(self, embed):
        embed.title = "🧠 AI Commands (Groq + Gemini)"
        embed.description = "AI features powered by Groq (fast text) and Gemini (?rate only)."
        embed.add_field(name="`?talk <message>`", value="Chat with the AI. Reply to its answer to keep chatting without retyping `?talk`. Use `?reset` to clear memory.", inline=False)
        embed.add_field(name="`?reset`", value="Clears your AI conversation memory.", inline=False)
        embed.add_field(name="`?rate [user]`", value="Gemini analyzes their actual avatar image: rating out of 5, description, and an improvement suggestion.", inline=False)
        embed.add_field(name="`?create`", value="Open the Poem/Riddle/Song creative-thread menu.", inline=False)
        embed.add_field(name="`?hpoem`", value="Get a poem hint (inside a Poem thread, max 3 hints).", inline=False)
        embed.add_field(name="`?suggesth` / `?suggeste`", value="Get Hindi / English song suggestions (inside a Song thread).", inline=False)
        return embed

    def get_fun_ai_page(self, embed):
        embed.title = "🎉 Fun AI Features"
        embed.description = "AI-powered stuff for laughs and catching up on chat."
        embed.add_field(name="`?vibecheck`", value="AI reads the recent chat and gives a fun read on the server's current vibe.", inline=False)
        embed.add_field(name="`?roast [user]`", value="AI roasts someone based on their actual activity. All in good fun!", inline=False)
        embed.add_field(name="`?wrapped [user]`", value="A 'Spotify Wrapped'-style recap of a user's time in the server.", inline=False)
        embed.add_field(name="`?tldr [minutes]`", value="AI summarizes recent chat (default: last 60 minutes) so you can catch up fast.", inline=False)
        embed.add_field(name="`?onthisday`", value="Resurfaces a random message from this exact day in a previous year, in this channel.", inline=False)
        embed.add_field(name="`?droll <count> <min>-<max>`", value="Rolls random numbers, then crowns whoever called one of them FIRST in recent chat.", inline=False)
        return embed

    def get_legends_page(self, embed):
        embed.title = "🎭 Legends & Lore"
        embed.description = "The server's own living memory and personality."
        embed.add_field(name="React with 📝", value="Saves that message to the server's memory bank -- it'll randomly resurface later, unprompted.", inline=False)
        embed.add_field(name="`?servermood`", value="Shows the server's current 'mood' -- drifts live based on recent chat energy.", inline=False)
        embed.add_field(name="`?evolution [user]`", value="AI 'character arc' narrative of someone's journey in the server, based on real data.", inline=False)
        embed.add_field(name="`?serverlore`", value="AI writes a fake 'wiki page' history of the server based on real stats.", inline=False)
        embed.add_field(name="`?card [user]`", value="Trading-card-style profile with stats, badges, and AI-generated art.", inline=False)
        embed.add_field(name="`?court @user`", value="Mock trial -- chat votes guilty/innocent, loser gets a silly 1-hour role.", inline=False)
        embed.add_field(name="`?globalrep [user]`", value="Reputation & badges aggregated across every server the bot shares with them.", inline=False)
        embed.add_field(name='`?timecapsule <DD/MM/YYYY> <HH:MM> <message>`', value="Seals a message that gets publicly revealed here at a future date.", inline=False)
        return embed

    def get_engagement_page(self, embed):
        embed.title = "🏆 Engagement & Fun"
        embed.description = "Roles, reputation, and leaderboards."
        embed.add_field(name="`?setup_roles`", value="Post the interactive self-role menu, built from this server's configured categories (admins only).", inline=False)
        embed.add_field(name="`?addnewcategory <name>`", value="Adds a brand-new category to the role menu (admins only). Re-run `?setup_roles` after.", inline=False)
        embed.add_field(name="`?addnewsetup <category> <role>`", value="Adds a new self-assignable role to a category -- auto-creates it (colourless, with an emoji) if it doesn't exist yet (admins only). Re-run `?setup_roles` after.", inline=False)
        embed.add_field(name="`?rep @user`", value="Give someone a reputation point (once per hour).", inline=False)
        embed.add_field(name="`?profile [user]`", value="View message count, reputation, and earned badges.", inline=False)
        embed.add_field(name="`?badges [user]`", value="See all achievement badges a user has earned.", inline=False)
        embed.add_field(name="`?leaderboard` (`?lb`)", value="Top 10 chatters this week.", inline=False)
        embed.add_field(name="`?wish @user`", value="Give a member a birthday role + announcement (mods only).", inline=False)
        embed.add_field(name="`?addbday <day> <month> <year>`", value="Register your birthday -- the bot auto-celebrates it at midnight IST with an AI-personalized announcement, DM, image, and a one-day custom role.", inline=False)
        embed.add_field(name="`?activatebday`", value="[Mods only] Instantly run the full birthday celebration for yourself right now, for demo/testing.", inline=False)
        embed.add_field(name="`?poll \"question\" opt1 opt2 ...`", value="Creates a reaction poll (up to 10 options).", inline=False)
        embed.add_field(name="`?suggest <idea>`", value="Submit a suggestion -- posted with 👍/👎 voting.", inline=False)
        embed.add_field(name="React with 📌", value="Any mod can pin a message just by reacting to it with 📌.", inline=False)
        embed.add_field(name="`?addtrigger \"phrase\" [text]`", value="[Mods only] Auto-reply with text/image whenever someone says that phrase. Attach an image too if you want. See also `?removetrigger`, `?listtriggers`.", inline=False)
        return embed

    def get_relationships_page(self, embed):
        embed.title = "💍 Marriage & Friendship"
        embed.description = "Silly server relationship system."
        embed.add_field(name="`?ship [user1] [user2]`", value="Compatibility %. 85%+ unlocks a Marry button (mutual consent, auto-handles existing marriages).", inline=False)
        embed.add_field(name="`?compatibility [user1] [user2]`", value="Same compatibility %/vibe check, no marry button -- just the read.", inline=False)
        embed.add_field(name="`?marry @user`", value="Propose marriage (they must accept).", inline=False)
        embed.add_field(name="`?divorce`", value="Divorces your current partner.", inline=False)
        embed.add_field(name="`?couple [user]`", value="Shows who someone is CURRENTLY married to (if anyone).", inline=False)
        embed.add_field(name="`?marriages`", value="All-time leaderboard of who's been married the most (includes past marriages -- not current status).", inline=False)
        embed.add_field(name="`?friend @user`", value="Send a friend request (they must accept).", inline=False)
        embed.add_field(name="`?unfriend @user`", value="Removes a friend.", inline=False)
        embed.add_field(name="`?friendslist [user]`", value="Shows a user's friends list.", inline=False)
        return embed

    def get_reminders_page(self, embed):
        embed.title = "⏰ Reminders & Scheduled DMs"
        embed.description = "Get pinged in your DMs later."
        embed.add_field(name='`?remind "<title>" <DD/MM/YYYY> <HH:MM>`', value="Set a public reminder for yourself.", inline=False)
        embed.add_field(name='`?remindpvt "<title>" <DD/MM/YYYY> <HH:MM>`', value="Set a private (hidden confirmation) reminder for yourself.", inline=False)
        embed.add_field(name='`?senddm <user> "<title>" <DD/MM/YYYY> <HH:MM>`', value="Schedule a DM to be sent to any user.", inline=False)
        return embed

    def get_private_channels_page(self, embed):
        embed.title = "🔒 Private Member Channels"
        embed.description = "Fully member-owned channels -- not gated by mod permissions. Only the creator controls access."
        embed.add_field(name="`?createpvtchannel <name>`", value=f"Creates a private channel only you (and the bot) can see, up to {MAX_PRIVATE_CHANNELS_PER_USER} per person.", inline=False)
        embed.add_field(name="`?allowin @user`", value="[Run inside your channel] Lets a user view it (read-only).", inline=False)
        embed.add_field(name="`?allowtype @user`", value="[Run inside your channel] Lets a user who's allowed in start typing.", inline=False)
        embed.add_field(name="`?disallowin @user`", value="[Run inside your channel] Removes a user's access entirely.", inline=False)
        embed.add_field(name="`?deletepvtchannel`", value="[Run inside your channel] Deletes it.", inline=False)
        embed.add_field(name="⚠️ Note", value="A true **Administrator** on the server can always see every channel -- that's a Discord platform limit, not something any bot can override.", inline=False)
        return embed

    def get_story_page(self, embed):
        embed.title = "📖 AI Roleplay Stories"
        embed.description = "An interactive AI roleplay story, played out live in a thread."
        embed.add_field(name="`?bnstory`", value="Starts a new story -- pick public/private, then genre(s) (Adventure, Romantic, Horror, Emotional, Fantasy, Fiction -- pick multiple to mix them), then a duration (`30m` to `2d`, minimum 30 minutes). The AI narrates and plays every other character; you play yourself by just typing in the thread.", inline=False)
        embed.add_field(name="`?stallowsee @user`", value="[Story creator only, private stories] Lets someone view your private story thread.", inline=False)
        embed.add_field(name="`?stallowwrite @user`", value="[Story creator only, private stories] Lets someone view AND write in your private story thread.", inline=False)
        embed.add_field(name="When time's up", value="The AI wraps the story up with a real ending, then asks if you want to start another one (same thread) or be done (thread auto-deletes).", inline=False)
        return embed

    def get_new_features_page(self, embed):
        embed.title = "✨ Unique Features"
        embed.description = "Stuff you won't find in most Discord bots."
        embed.add_field(name="`?familytree [user]`", value="Visual diagram of marriages (red) and friendships (green) -- server-wide, or just one person's connections.", inline=False)
        embed.add_field(name="`?mediate @user1 @user2`", value="AI reads the recent back-and-forth between two people and gives a neutral summary + lighthearted verdict.", inline=False)
        embed.add_field(name="`?twin [user]`", value="Finds who you interact with the most -- voice time together, replies, and being active in chat at the same time.", inline=False)
        embed.add_field(name="`?confess <message>`", value="Posts your message anonymously in the confessions channel -- AI-screened for safety first. Works in a server channel, or DM the bot directly. Requires a mod to run `?setchannel confessions #channel` once.", inline=False)
        embed.add_field(name="Auto-Eulogy", value="When someone leaves for good, an AI-written send-off based on their real stats posts automatically -- requires `?setchannel eulogy #channel`.", inline=False)
        embed.add_field(name="Weekly Digest DM", value="Every active member gets their own AI-written personal recap DMed to them when the weekly leaderboard resets. Opt out with `?digestoptout` (back in with `?digestoptin`).", inline=False)
        embed.add_field(name="Random Ambient Roast", value="Every 6 hours, on every server independently, the bot picks a random recent chatter in the most active channel and posts a savage, unprompted AI roast grounded in their real messages. No per-user opt-out, but mods can turn it off server-wide with `?disableroast`.", inline=False)
        embed.add_field(name="True Global Rarity", value="`?badges` now shows how many people across EVERY server this bot is in have ever earned each badge -- not just this server.", inline=False)
        return embed

    def get_games_page(self, embed):
        embed.title = "🎉 Fun & Games"
        embed.description = "Pure entertainment commands."
        embed.add_field(name="`?awards`", value="AI-generated server awards show based on real activity stats (chattiest, VC MVP, most married, etc).", inline=False)
        embed.add_field(name="`?impersonate` / `?guessimpersonate @user`", value="AI mimics a random active member's writing style -- guess who it is!", inline=False)
        embed.add_field(name="`?horoscope [sign]`", value="Daily AI horoscope themed around this server.", inline=False)
        embed.add_field(name="`?tarot [user]`", value="AI tarot reading woven around your real server stats.", inline=False)
        embed.add_field(name="`?roastbattle @user1 @user2`", value="AI roasts two people head-to-head and declares a winner.", inline=False)
        embed.add_field(name="`?roulette`", value="Spin for a random harmless consequence -- timeout, silly nickname, color role, or nothing.", inline=False)
        embed.add_field(name="`?datingshow`", value="AI picks 3 random unmarried members and generates a silly 'who would you pick' scenario.", inline=False)
        embed.add_field(name="`?cursedcombo [user1] [user2]`", value="Mashes two members + a random scenario into an absurd AI short story.", inline=False)
        embed.add_field(name="`?captionbattle` / `?submitcaption <text>`", value="Bot posts a weird AI image -- submit captions, then chat votes on the winner.", inline=False)
        embed.add_field(name="`?wyr`", value="AI-generated 'Would You Rather' -- react 🅰️ or 🅱️ to vote, hit Next Question to keep going.", inline=False)
        embed.add_field(name="`?truthordare` (`?tod`) `[@user]`", value="Truth or Dare -- the target picks Truth or Dare via buttons, AI generates one, and a Next button keeps the round going.", inline=False)
        embed.add_field(name="`?whowouldyou` (`?wwy`)", value="Picks two random real members of this server -- react 🅰️ or 🅱️ to vote for who you'd choose. Whoever gets fewer votes gets a cursed role for 24 hours. Keep going with Next Round.", inline=False)
        return embed

    def get_owner_page(self, embed):
        embed.title = "👑 Owner Commands"
        embed.description = "Exclusive for **kanjuubarfiiii** and **huh.ashh**."
        embed.add_field(name="`?online` / `?idle` / `?dnd`", value="Change the bot's status icon.", inline=False)
        return embed

    def get_bot_identity_page(self, embed):
        embed.title = "🤖 Bot Identity (Admins only)"
        embed.description = "Change the bot's actual profile -- no dev portal needed. ⚠️ These are global: they affect the bot in every server it's in, not just this one."
        embed.add_field(name="`?setpfp <url>` (or attach an image)", value="Change the bot's avatar.", inline=False)

        embed.add_field(name="`?setactivity <playing|watching|listening|competing> <text>`", value="Change what shows next to the bot's name.", inline=False)
        embed.add_field(name="`?setabout <text>`", value="Change the bot's 'About Me' description (max 400 chars).", inline=False)
        return embed

    def get_server_config_page(self, embed):
        embed.title = "⚙️ Server Setup (Mods only)"
        embed.description = "Configure which channel each feature posts to, ON THIS SERVER specifically. Needed once per server -- these features can't guess the right channel on their own."
        embed.add_field(
            name='`?setchannel <birthday|leaderboard|suggestions|welcome|modlog|ghostlog|eulogy|confessions> [#channel]`',
            value="Sets the channel for that feature. Omit the channel to use the one you're typing in. `welcome` = join messages, `modlog` = warn/kick/ban/etc logs, `ghostlog` = deleted/edited ping alerts, `eulogy` = auto-posted send-off when someone leaves, `confessions` = anonymous confession posts.",
            inline=False
        )
        embed.add_field(name="`?triggerchannel \"phrase\" [#channel]`", value="Restricts a trigger to only fire in one channel (omit channel to clear).", inline=False)
        embed.add_field(name="`?triggertransfer #source #destination` / `?removetriggertransfer #source`", value="Triggers typed in #source post their response in #destination instead.", inline=False)
        embed.add_field(name="`?restricttalk #channel` / `?unrestricttalk #channel`", value="Whitelist which channels `?talk` can be used in. If none whitelisted, it works everywhere.", inline=False)
        embed.add_field(name="`?listtalkchannels`", value="Shows the current `?talk` channel whitelist.", inline=False)
        embed.add_field(name="`?setjailchannel #channel`", value="Sets the jail channel and locks the Jailed role out of every other channel (text + voice). Set the channel's own visibility (hidden from regular members, visible to mods) yourself first.", inline=False)
        embed.add_field(name="`?setcontrolchannel #channel`", value="[Admins only] Designates a channel where plain-English instructions (e.g. `timeout @user for 1 hour`, `unjail @user after 10 mins`) get carried out automatically by an AI parser.", inline=False)
        embed.add_field(name="`?disableroast` / `?enableroast`", value="Turns the automatic ~6-hourly random roast off/on for this server.", inline=False)
        return embed

    @ui.button(label="<", style=discord.ButtonStyle.primary)
    async def prev(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user != self.author:
            return await interaction.response.send_message("This isn't your menu!", ephemeral=True)
        self.current_page = max(1, self.current_page - 1)
        await interaction.response.edit_message(embed=self.create_page(self.current_page), view=self)

    @ui.button(label=">", style=discord.ButtonStyle.primary)
    async def next(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user != self.author:
            return await interaction.response.send_message("This isn't your menu!", ephemeral=True)
        self.current_page = min(len(self.pages), self.current_page + 1)
        await interaction.response.edit_message(embed=self.create_page(self.current_page), view=self)

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass

@bot.command()
async def help(ctx):
    """Displays the interactive, multi-page help guide."""
    view = HelpView(bot, ctx.author)
    embed = view.create_page(1)
    view.message = await ctx.send(embed=embed, view=view)


# --------------------------------------------------------
# ❌ ERROR HANDLING
# --------------------------------------------------------

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return 
    
    if isinstance(error, commands.MissingRequiredArgument):
        return await ctx.send(f"❌ **Missing Argument:** You are missing a required argument for this command. Usage: `{bot.command_prefix}{ctx.command.name} {ctx.command.usage or ''}`", delete_after=15)
        
    if isinstance(error, commands.BadArgument):
        if ctx.command.name == 'senddm' and isinstance(error, commands.UserNotFound):
            return await ctx.send("❌ **Recipient Error:** Could not find that user. Please ensure you are mentioning a user or providing a valid User ID.", delete_after=15)
            
        return await ctx.send(f"❌ **Invalid Argument:** Please check the type of argument you provided (e.g., mention a user, use an integer, use correct date format). Usage: `{bot.command_prefix}{ctx.command.name} {ctx.command.usage or ''}`", delete_after=15)

    if isinstance(error, commands.MissingPermissions):
        permission_list = [p.replace('_', ' ').title() for p in error.missing_permissions]
        return await ctx.send(f"❌ **Permission Denied:** You need the following permission(s) to use this: `{', '.join(permission_list)}`", delete_after=15)

    if isinstance(error, commands.BotMissingPermissions):
        permission_list = [p.replace('_', ' ').title() for p in error.missing_permissions]
        return await ctx.send(f"❌ **Bot Permission Error:** I need the following permission(s) to execute this: `{', '.join(permission_list)}`")

    if isinstance(error, commands.CommandInvokeError):
        original = error.original

        if isinstance(original, discord.HTTPException) and original.status == 429:
            print(f"🚨 Rate limited (429) while running {ctx.command}: {original}")
            return await ctx.send(
                "⏳ **Rate limited by Discord.** The bot has been making requests too quickly "
                "and Discord is temporarily throttling it. Wait a minute or two and try again.",
                delete_after=20
            )

        print(f"🚨 Unhandled CommandInvokeError in command {ctx.command}: {original}")
        return await ctx.send(f"❌ An internal error occurred while running this command. The developer has been notified (Error type: {type(original).__name__}).", delete_after=15)


    print(f"Ignoring unhandled exception in command {ctx.command}: {error}")

# --- BOT RUNNER ---
try:
    TOKEN = os.getenv("DISCORD_TOKEN")
    bot.run(TOKEN)
except discord.errors.LoginFailure:
    print("\n\nFATAL ERROR: Improper token has been passed. Check your TOKEN variable.")
except Exception as e:

    print(f"\n\nAn unexpected error occurred: {e}")
