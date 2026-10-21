"""Провайдеры нейросети для настроек: рекомендуемые модели, где взять ключ, проверка связи."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from assistant.config import LlmConfig, LlmProfile
from assistant.llm.client import LlmError, create_client

log = logging.getLogger(__name__)

# Свой OpenAI-совместимый сервис: профиль создаётся из настроек интерфейса.
CUSTOM_PROFILE = "custom"
CUSTOM_TITLE = "Другой сервис"
CUSTOM_KEY_ENV = "CUSTOM_LLM_API_KEY"


@dataclass(frozen=True)
class ProviderInfo:
    """Что показать в настройках о провайдере."""

    models: tuple[str, ...] = ()  # рекомендуемые модели
    key_url: str = ""  # где создать ключ
    note: str = ""  # пояснение; названия моделей — в `обратных кавычках`
    base_url_editable: bool = False


PROVIDERS: dict[str, ProviderInfo] = {
    "deepseek": ProviderInfo(
        models=("deepseek-flash",),
        key_url="https://platform.deepseek.com/api_keys",
        note="Быстрая и недорогая. Ключ работает, только если на счёте есть деньги.",
    ),
    # Только модели, которые в Chat Completions вызывают инструменты (с reasoning_effort "none").
    # GPT-6 Astra и GPT-6.1 Sol делают это лишь через Responses API — их здесь нет.
    "openai": ProviderInfo(
        models=("gpt-6-luna", "gpt-5.4-mini", "gpt-6-sol", "gpt-5.5"),
        key_url="https://platform.openai.com/settings/organization/api-keys",
        note="Модели OpenAI. В списке те, что умеют вызывать действия; быстрее всех — `gpt-6-luna`.",
    ),
    "anthropic": ProviderInfo(
        models=("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
        key_url="https://platform.claude.com/settings/keys",
        note=(
            "Модели Anthropic. Сложные задачи лучше всех решает `claude-opus-5-5`, "
            "быстрее всех отвечает `claude-haiku-5-5`."
        ),
    ),
    "ollama": ProviderInfo(
        models=("qwen3:8b",),
        note=(
            "Работает без интернета на этом компьютере. Нужна установленная Ollama "
            "и скачанная модель: `ollama pull qwen3:8b`."
        ),
        base_url_editable=True,
    ),
    "opencode_zen": ProviderInfo(
        key_url="https://opencode.ai/zen",
        note=(
            "Шлюз OpenCode к моделям разных компаний. Бесплатные модели Zen отвечают "
            "только приложению OpenCode — здесь нужна платная."
        ),
    ),
    CUSTOM_PROFILE: ProviderInfo(
        note="Любой сервис с OpenAI-совместимым API: укажите адрес, модель и ключ, если он нужен.",
        base_url_editable=True,
    ),
}


def provider_info(name: str, profile: LlmProfile | None = None) -> ProviderInfo:
    """Сведения о профиле; для профилей, добавленных в config.yaml вручную, — общие."""
    if name in PROVIDERS:
        return PROVIDERS[name]
    return ProviderInfo(
        note="Профиль из config.yaml.",
        base_url_editable=profile is not None and profile.provider == "openai",
    )


def build_profile(
    settings: LlmConfig, name: str, model: str, base_url: str = "", key_env: str = ""
) -> LlmProfile:
    """Профиль с выбранной в настройках моделью (и адресом, если его можно менять).

    Для своего сервиса (custom), которого ещё нет в конфиге, профиль создаётся.

    Raises:
        pydantic.ValidationError: пустая модель, нет адреса и т. п.
    """
    current = settings.profiles.get(name)
    if current is None:
        if name != CUSTOM_PROFILE:
            raise KeyError(name)
        return LlmProfile.model_validate(
            {"title": CUSTOM_TITLE, "base_url": base_url, "model": model, "api_key_env": key_env}
        )
    data = current.model_dump()
    data["model"] = model
    if base_url and provider_info(name, current).base_url_editable:
        data["base_url"] = base_url
    if name == CUSTOM_PROFILE and key_env:
        data["api_key_env"] = key_env
    return LlmProfile.model_validate(data)


def with_profile(settings: LlmConfig, name: str, profile: LlmProfile) -> LlmConfig:
    """Раздел llm, где выбран и заменён профиль name (для проверки до сохранения)."""
    return settings.model_copy(update={"profile": name, "profiles": {**settings.profiles, name: profile}})


@dataclass(frozen=True)
class CheckResult:
    """Итог проверки связи."""

    ok: bool
    text: str = ""
    ms: float = 0.0
    error: str = ""


# Инструмент в проверке нужен: часть моделей отвечает текстом, но отказывает в инструментах.
_PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Проверка связи. Не вызывай этот инструмент.",
        "parameters": {"type": "object", "properties": {}},
    },
}
_PING_MESSAGES = [
    {"role": "system", "content": "Это проверка связи. Отвечай одним словом."},
    {"role": "user", "content": "Ответь словом «готово»."},
]


def check_connection(settings: LlmConfig, api_key: str | None = None, timeout: float = 30) -> CheckResult:
    """Короткий запрос к выбранному профилю — такой же, как у агента, с инструментом.

    Args:
        settings: раздел llm с выбранным профилем.
        api_key: ключ, который ещё не сохранён; None — из .env.
        timeout: лимит на запрос, секунды.
    """
    try:
        client = create_client(settings, api_key=api_key)
        reply = client.complete(_PING_MESSAGES, [_PING_TOOL], timeout=timeout)
    except LlmError as exc:
        log.info("Проверка нейросети %s: %s", settings.profile, exc)
        return CheckResult(False, error=str(exc))
    text = reply.text.strip() or ("вызвала инструмент" if reply.tool_calls else "")
    log.info("Проверка нейросети %s: ответ за %.0f мс", settings.profile, reply.total_ms)
    return CheckResult(True, text, reply.total_ms)


def list_models(settings: LlmConfig, api_key: str | None = None) -> list[str]:
    """Модели, доступные по ключу.

    Raises:
        LlmError: провайдер не ответил или отказал.
    """
    return sorted(create_client(settings, api_key=api_key).list_models())
