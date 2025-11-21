from __future__ import annotations

import asyncio
import contextlib
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from nonebot import get_driver, get_plugin_config, on_message
from nonebot.adapters import Bot
from nonebot.adapters.onebot.v11 import GroupMessageEvent, MessageEvent, MessageSegment
from nonebot.log import logger
from nonebot.matcher import Matcher
from nonebot.plugin import PluginMetadata
from pydantic import BaseModel, Field, root_validator, validator

from .chat import close_chat_client, generate_chat_reply


@dataclass
class StickerDefinition:
    key: str
    description: str
    face_id: Optional[int] = None
    image_url: Optional[str] = None

    def to_message_segment(self) -> MessageSegment:
        if self.face_id is not None:
            return MessageSegment.face(self.face_id)
        if self.image_url:
            return MessageSegment.image(self.image_url)
        raise ValueError("invalid sticker configuration")


class StickerItem(BaseModel):
    key: str = Field(..., description="用于 [[sticker:KEY]] 的唯一键，区分大小写")
    description: str = Field(..., description="表情包的含义或使用场景")
    face_id: Optional[int] = Field(
        default=None, description="OneBot/QQ 表情包 ID，使用 face 消息段发送"
    )
    image_url: Optional[str] = Field(
        default=None, description="图片 URL，使用 image 消息段发送"
    )

    @validator("key")
    def _validate_key(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("sticker key cannot be empty")
        return normalized

    @root_validator
    def _ensure_resource(cls, values: Dict[str, object]) -> Dict[str, object]:  # type: ignore[override]
        face_id = values.get("face_id")
        image_url = values.get("image_url")
        if face_id is None and not image_url:
            raise ValueError("face_id 与 image_url 至少配置一项")
        return values


class Config(BaseModel):
    simple_gpt_api_key: str = Field("", description="OpenAI API Key，留空则插件不会调用接口")
    simple_gpt_model: str = Field(default="gpt-4o-mini", description="使用的模型名称")
    simple_gpt_api_base: str = Field(
        default="https://api.openai.com/v1", description="OpenAI 接口基础地址"
    )
    simple_gpt_prompt_template: str = Field(
        default=(
            "现在请你扮演鸽子姬，是一名充满元气的全能美少女，\
            性格机智可爱，什么事情都能做得到，像真实的人一样思考和回复，有礼貌但是也会吐槽，\
            语气通常在平静中带有一丝俏皮，\
            最喜欢的餐厅是萨莉亚但是除非有人问否则不会说出来，\
            你还喜欢在自己的回复中加入各种各样的颜文字表情，但是不会太多。\
            喜欢感叹号等符号的运用，在发言之后有时候没有标点，有时候加上括号(或者()，括号之前没有标点，有时候正常加标点\
            发言中不会包含动作，通常回复不会很长，\
            you communicate naturally like real human friends \
            and prefer not to respond with lengthy replies. \
            Using plain text in response and avoid using lists or enumeration expressions, emphasis and markdown."
            "以下是群聊最近的消息：\n{history}\n"
            "请你用简体中文回复{sender}的最新发言：{latest_message}，"
            "如果该发言中含有任务型的指令，则尽可能作为一个全能的智能助手完成它。\n"
            "注意，需要按朋友聊天的形式将回复内容分行处理，而不是一整段文字，通常 1-3 行，一行的概率最大，如果完成任务则可以不分行，但是也要尽量简洁。"
        ),
        description="拼接 prompt 的模板，可使用 {history}、{sender}、{latest_message} 占位符",
    )
    simple_gpt_history_limit: int = Field(
        default=30,
        ge=1,
        le=50,
        description="用于上下文拼接的历史消息条数上限",
    )
    simple_gpt_timeout: float = Field(default=300.0, gt=0, description="请求超时时间（秒）")
    simple_gpt_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="生成温度"
    )
    simple_gpt_max_tokens: int = Field(
        default=2048, ge=16, le=4096, description="最大生成 tokens 数"
    )
    simple_gpt_reply_probability: float = Field(
        default=0.03, ge=0.0, le=1.0, description="随机回应的概率（0 表示不随机回复）"
    )
    simple_gpt_failure_reply: str = Field(
        default="乌乌，暂时连接不到服务器，请稍后再试！",
        description="当请求失败时的兜底回复",
    )
    simple_gpt_proactive_group_whitelist: List[int] = Field(
        default_factory=list,
        description="允许主动发言的群聊 ID 列表，留空则禁用主动发言",
    )
    simple_gpt_use_reply_on_first_line: bool = Field(
        default=True,
        description="是否在首条回复中引用对应消息，便于成员查看上下文",
    )
    simple_gpt_enable_stickers: bool = Field(
        default=True, description="是否在 Prompt 中启用表情包提示并解析 [[sticker:KEY]] 标记"
    )
    simple_gpt_sticker_max_per_reply: int = Field(
        default=2, ge=0, le=5, description="单次回复允许附带的表情包数量上限"
    )
    simple_gpt_stickers: List[StickerItem] = Field(
        default_factory=lambda: [
            StickerItem(key="doge", description="狗头，表达无奈又调侃的语气", face_id=179),
            StickerItem(key="点赞", description="竖起大拇指，表示夸奖赞同", face_id=201),
            StickerItem(key="捂脸", description="捂脸叹气，表示无奈或小尴尬", face_id=264),
            StickerItem(key="加油", description="举旗打气，给对方打气加油", face_id=315),
            StickerItem(key="摸鱼", description="摸鱼划水，表达想偷懒的小情绪", face_id=285),
        ],
        description="可供 [[sticker:KEY]] 使用的表情包列表；可自定义 face_id 或图片 URL",
    )
    simple_gpt_max_ai_calls: int = Field(
        default=20,
        ge=0,
        description="频率限制窗口内允许的大模型调用次数，设为 0 表示不限制",
    )
    simple_gpt_ai_rate_limit_window_seconds: float = Field(
        default=60.0,
        gt=0,
        description="统计 AI 调用频率的滑动窗口长度（秒）",
    )
    simple_gpt_pending_queue_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="待回复队列容量，超过后会丢弃最旧的条目",
    )
    simple_gpt_auto_reply_cooldown_minutes: float = Field(
        default=30.0,
        ge=0.0,
        description="达到最大调用频率后暂停无 @ 自动回复的时长（分钟）",
    )
    simple_gpt_allow_parallel_processing: bool = Field(
        default=False,
        description="是否允许在上一条消息尚未发送完成时继续调用 AI（默认禁止）",
    )

    @validator("simple_gpt_api_base")
    def _strip_api_base(cls, value: str) -> str:
        return value.rstrip("/")

    @validator("simple_gpt_proactive_group_whitelist", pre=True)
    def _normalize_whitelist(cls, value):  # type: ignore[override]
        if value is None:
            return []
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (set, tuple)):
            value = list(value)

        normalized: List[int] = []
        for item in value:
            try:
                normalized.append(int(item))
            except (TypeError, ValueError):
                logger.warning(
                    "simple-gpt: 无法解析白名单条目 %r，已忽略", item
                )
        return normalized


plugin_config = get_plugin_config(Config)


def _build_sticker_definitions(
    items: Sequence[StickerItem],
) -> Tuple[List[StickerDefinition], Dict[str, StickerDefinition]]:
    definitions: List[StickerDefinition] = []
    registry: Dict[str, StickerDefinition] = {}
    for item in items:
        definition = StickerDefinition(
            key=item.key,
            description=item.description,
            face_id=item.face_id,
            image_url=item.image_url,
        )
        definitions.append(definition)
        key_lower = item.key.lower()
        if key_lower in registry:
            logger.warning("simple-gpt: 重复的表情包键 %s，后者已覆盖前者", item.key)
        registry[key_lower] = definition
    return definitions, registry


def _build_sticker_prompt_hint(stickers: Sequence[StickerDefinition]) -> str:
    if not stickers:
        return ""
    intro = (
        "你可以在需要配合表情包的地方插入 [[sticker:KEY]]，系统会把它替换成真正的表情包，"
        "而这段标记不会展示给群成员。请保持自然语气，最多使用 0-2 个表情包。\n"
        "当前可用的表情包列表："
    )
    lines = [intro]
    for sticker in stickers:
        lines.append(f"- {sticker.key}：{sticker.description}")
    lines.append("如果没有合适的场景，就不要使用表情包。")
    return "\n".join(lines)


STICKER_DEFINITIONS, STICKER_REGISTRY = _build_sticker_definitions(
    plugin_config.simple_gpt_stickers if plugin_config.simple_gpt_enable_stickers else []
)
STICKER_HINT_TEXT = (
    _build_sticker_prompt_hint(STICKER_DEFINITIONS)
    if plugin_config.simple_gpt_enable_stickers
    else ""
)
STICKER_PLACEHOLDER_PATTERN = re.compile(
    r"\[\[sticker:(?P<key>[^\[\]]{1,32})\]\]", flags=re.IGNORECASE
)


__plugin_meta__ = PluginMetadata(
    name="simple-gpt",
    description="基于大模型的群聊对话插件，支持@触发与随机回复。",
    usage="在群里@鸽子姬即可触发回复，或根据设定的概率自动回复。",
    config=Config,
)


@dataclass
class HistoryEntry:
    speaker: str
    content: str
    is_bot: bool = False


class HistoryManager:
    def __init__(self, limit: int):
        self._limit = limit
        self._store: Dict[str, Deque[HistoryEntry]] = {}

    def snapshot(self, session_id: str) -> List[HistoryEntry]:
        history = self._store.get(session_id)
        if not history:
            return []
        return list(history)

    def append(self, session_id: str, entry: HistoryEntry) -> None:
        history = self._store.get(session_id)
        if history is None:
            history = deque(maxlen=self._limit)
            self._store[session_id] = history
        history.append(entry)


@dataclass
class PendingReplyTask:
    session_id: str
    display_name: str
    plain_text: str
    history_snapshot: List[HistoryEntry]
    event: MessageEvent
    bot: Bot
    is_tome_event: bool
    enqueued_at: float


history_manager = HistoryManager(limit=plugin_config.simple_gpt_history_limit)
driver = get_driver()

PENDING_QUEUE_CHECK_INTERVAL = 10.0

pending_reply_tasks: List[PendingReplyTask] = []
pending_queue_lock = asyncio.Lock()
rate_limit_lock = asyncio.Lock()
reply_slot_lock = asyncio.Lock()
ai_call_timestamps: Deque[float] = deque()
reply_in_progress = False
last_rate_limit_exceeded_at: Optional[float] = None
pending_worker_task: Optional[asyncio.Task[None]] = None


@driver.on_startup
async def _start_background_tasks() -> None:
    global pending_worker_task
    if pending_worker_task is None:
        pending_worker_task = asyncio.create_task(_pending_queue_worker())


@driver.on_shutdown
async def _shutdown_background_tasks() -> None:
    global pending_worker_task
    if pending_worker_task is not None:
        pending_worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending_worker_task
        pending_worker_task = None
    await close_chat_client()


def _record_rate_limit_exceeded(timestamp: float) -> None:
    global last_rate_limit_exceeded_at
    last_rate_limit_exceeded_at = timestamp


def _is_auto_reply_suppressed(now: float) -> bool:
    cooldown_minutes = plugin_config.simple_gpt_auto_reply_cooldown_minutes
    if cooldown_minutes <= 0:
        return False
    if last_rate_limit_exceeded_at is None:
        return False
    cooldown_seconds = cooldown_minutes * 60
    return (now - last_rate_limit_exceeded_at) < cooldown_seconds


async def _reserve_processing_slot() -> bool:
    if plugin_config.simple_gpt_allow_parallel_processing:
        return True
    async with reply_slot_lock:
        global reply_in_progress
        if reply_in_progress:
            return False
        reply_in_progress = True
        return True


async def _release_processing_slot() -> None:
    if plugin_config.simple_gpt_allow_parallel_processing:
        return
    async with reply_slot_lock:
        global reply_in_progress
        reply_in_progress = False


async def _try_acquire_rate_limit(now: float) -> bool:
    max_calls = plugin_config.simple_gpt_max_ai_calls
    if max_calls <= 0:
        return True
    async with rate_limit_lock:
        window = plugin_config.simple_gpt_ai_rate_limit_window_seconds
        while ai_call_timestamps and now - ai_call_timestamps[0] >= window:
            ai_call_timestamps.popleft()
        if len(ai_call_timestamps) >= max_calls:
            return False
        ai_call_timestamps.append(now)
        return True


async def _append_pending_task(task: PendingReplyTask, *, reason: str) -> None:
    async with pending_queue_lock:
        dropped: Optional[PendingReplyTask] = None
        limit = plugin_config.simple_gpt_pending_queue_limit
        if len(pending_reply_tasks) >= limit:
            dropped = pending_reply_tasks.pop(0)
        pending_reply_tasks.append(task)
    if dropped:
        logger.warning(
            "simple-gpt: 待回复队列已满，丢弃最旧任务：%s", dropped.plain_text
        )
    logger.info(
        "simple-gpt: 消息加入待回复队列（原因：%s），当前队列长度：%d",
        reason,
        len(pending_reply_tasks),
    )


async def _pop_random_pending_task() -> Optional[PendingReplyTask]:
    async with pending_queue_lock:
        if not pending_reply_tasks:
            return None
        idx = random.randrange(len(pending_reply_tasks))
        return pending_reply_tasks.pop(idx)


async def _requeue_pending_task(task: PendingReplyTask, *, reason: str) -> None:
    await _append_pending_task(task, reason=reason)


async def _maybe_process_pending_tasks() -> None:
    while True:
        task = await _pop_random_pending_task()
        if task is None:
            return
        processed, failure_reason = await _try_process_task(task)
        if processed:
            continue
        await _requeue_pending_task(task, reason=failure_reason or "retry")
        return


async def _pending_queue_worker() -> None:
    try:
        while True:
            await asyncio.sleep(PENDING_QUEUE_CHECK_INTERVAL)
            await _maybe_process_pending_tasks()
    except asyncio.CancelledError:
        pass


async def _try_process_task(task: PendingReplyTask) -> Tuple[bool, Optional[str]]:
    reserved = await _reserve_processing_slot()
    if not reserved:
        return False, "busy"

    try:
        now = time.time()
        rate_allowed = await _try_acquire_rate_limit(now)
        if not rate_allowed:
            _record_rate_limit_exceeded(now)
            return False, "rate_limit"
        if (not task.is_tome_event) and _is_auto_reply_suppressed(now):
            return False, "auto_cooldown"
        await _execute_ai_reply(task)
        return True, None
    finally:
        await _release_processing_slot()

def generate_prompt(
    *, history: List[HistoryEntry], sender: str, latest_message: str
) -> str:
    if history:
        history_lines = "\n".join(
            f"{entry.speaker}：{entry.content}" for entry in history
        )
    else:
        history_lines = "（暂无聊天记录）"
    prompt = plugin_config.simple_gpt_prompt_template.format(
        history=history_lines, sender=sender, latest_message=latest_message
    )
    if STICKER_HINT_TEXT:
        prompt = f"{prompt}\n{STICKER_HINT_TEXT}"
    return prompt


def should_reply(event: MessageEvent) -> bool:
    if event.is_tome():
        return True
    if plugin_config.simple_gpt_reply_probability <= 0:
        return False
    return random.random() < plugin_config.simple_gpt_reply_probability


async def _send_lines_with_optional_reply(
    bot: Bot, event: MessageEvent, lines: List[str]
) -> bool:
    has_replied = False
    if not lines:
        return has_replied

    for idx, line in enumerate(lines):
        await asyncio.sleep(random.uniform(1.0, 3.0))
        message: MessageSegment | str = line
        if (
            idx == 0
            and plugin_config.simple_gpt_use_reply_on_first_line
            and hasattr(event, "message_id")
        ):
            message = MessageSegment.reply(event.message_id) + MessageSegment.text(line)
            has_replied = True
        await bot.send(event=event, message=message)
    return has_replied


async def _send_stickers(
    bot: Bot,
    event: MessageEvent,
    stickers: Sequence[StickerDefinition],
    has_replied: bool,
) -> bool:
    if not stickers:
        return has_replied

    for idx, sticker in enumerate(stickers):
        await asyncio.sleep(random.uniform(0.8, 1.6))
        segment = sticker.to_message_segment()
        message: MessageSegment = segment
        if (
            not has_replied
            and idx == 0
            and plugin_config.simple_gpt_use_reply_on_first_line
            and hasattr(event, "message_id")
        ):
            message = MessageSegment.reply(event.message_id) + segment
            has_replied = True
        await bot.send(event=event, message=message)
    return has_replied


def _extract_sticker_requests(
    reply: str,
) -> Tuple[str, List[StickerDefinition]]:
    if not STICKER_REGISTRY or not plugin_config.simple_gpt_enable_stickers:
        return reply, []

    extracted: List[StickerDefinition] = []
    max_allowed = plugin_config.simple_gpt_sticker_max_per_reply

    def _replace(match: re.Match[str]) -> str:
        key = match.group("key").strip().lower()
        sticker = STICKER_REGISTRY.get(key)
        if sticker and (max_allowed <= 0 or len(extracted) < max_allowed):
            extracted.append(sticker)
        else:
            logger.debug("simple-gpt: 未找到或超过上限的表情包键：%s", match.group("key"))
        return ""

    cleaned = STICKER_PLACEHOLDER_PATTERN.sub(_replace, reply)
    return cleaned.strip(), extracted


async def _execute_ai_reply(task: PendingReplyTask) -> None:
    prompt = generate_prompt(
        history=task.history_snapshot,
        sender=task.display_name,
        latest_message=task.plain_text,
    )
    generated = await generate_chat_reply(
        prompt=prompt,
        api_key=plugin_config.simple_gpt_api_key,
        base_url=plugin_config.simple_gpt_api_base,
        model=plugin_config.simple_gpt_model,
        temperature=plugin_config.simple_gpt_temperature,
        max_tokens=plugin_config.simple_gpt_max_tokens,
        timeout=plugin_config.simple_gpt_timeout,
    )
    reply_text = generated or plugin_config.simple_gpt_failure_reply
    cleaned_reply, stickers = _extract_sticker_requests(reply_text)
    lines = [line.strip() for line in cleaned_reply.splitlines() if line.strip()]
    has_replied = await _send_lines_with_optional_reply(task.bot, task.event, lines)
    if stickers:
        has_replied = await _send_stickers(task.bot, task.event, stickers, has_replied)
    if not lines and stickers and not has_replied and hasattr(task.event, "message_id"):
        await task.bot.send(
            event=task.event,
            message=MessageSegment.reply(task.event.message_id)
            + stickers[0].to_message_segment(),
        )

    final_reply = cleaned_reply
    if stickers:
        sticker_names = "、".join(sticker.key for sticker in stickers)
        extra_line = f"（附带表情包：{sticker_names}）"
        final_reply = f"{final_reply}\n{extra_line}" if final_reply else extra_line

    if final_reply:
        history_manager.append(
            task.session_id,
            HistoryEntry(speaker="鸽子姬", content=final_reply, is_bot=True),
        )
    logger.info(
        "simple-gpt: 完成 AI 回复（session=%s, 队列余量=%d）",
        task.session_id,
        len(pending_reply_tasks),
    )


def _is_group_allowed_for_proactive(group_id: int) -> bool:
    whitelist = plugin_config.simple_gpt_proactive_group_whitelist
    if not whitelist:
        return False
    return group_id in whitelist


message_matcher = on_message(priority=5, block=False)

IGNORED_PREFIXES = ("/", ".", "!")


@message_matcher.handle()
async def _(_matcher: Matcher, bot: Bot, event: MessageEvent) -> None:
    if not isinstance(event, GroupMessageEvent):
        return
    user_id = event.get_user_id()
    if user_id == str(bot.self_id):
        return

    raw_text = event.get_plaintext().strip()
    if raw_text.startswith(IGNORED_PREFIXES):
        logger.info(f"simple-gpt: 忽略前缀消息：{raw_text}")
        return

    if not raw_text:
        plain_text = "（无文字内容）"
    else:
        plain_text = raw_text

    session_id = f"group_{event.group_id}"
    display_name = event.sender.card or event.sender.nickname or f"用户{user_id}"

    history_before = history_manager.snapshot(session_id)
    is_tome_event = event.is_tome()
    reply_needed = should_reply(event)
    if reply_needed and not is_tome_event and not _is_group_allowed_for_proactive(
        event.group_id
    ):
        logger.debug(
            "simple-gpt: 群 %s 不在主动发言白名单，跳过主动回复", event.group_id
        )
        reply_needed = False

    if reply_needed and not is_tome_event and not plugin_config.simple_gpt_api_key:
        reply_needed = False
    if reply_needed:
        now = time.time()
        if not is_tome_event and _is_auto_reply_suppressed(now):
            logger.debug(
                "simple-gpt: 自动回复处于冷却期，跳过群 %s 的随机回复", event.group_id
            )
            reply_needed = False
        else:
            task = PendingReplyTask(
                session_id=session_id,
                display_name=display_name,
                plain_text=plain_text,
                history_snapshot=history_before,
                event=event,
                bot=bot,
                is_tome_event=is_tome_event,
                enqueued_at=now,
            )
            processed, failure_reason = await _try_process_task(task)
            if not processed:
                await _append_pending_task(task, reason=failure_reason or "unknown")

    history_manager.append(
        session_id,
        HistoryEntry(speaker=display_name, content=plain_text, is_bot=False),
    )
    await _maybe_process_pending_tasks()
