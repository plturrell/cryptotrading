"""
AI Translation Service for Multi-Language Support
Supports English and Russian with Claude AI integration
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Language(Enum):
    ENGLISH = "en"
    RUSSIAN = "ru"
    CHINESE = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    KOREAN = "ko"


@dataclass
class TranslationRequest:
    text: str
    source_lang: Language
    target_lang: Language
    context: Optional[str] = None
    domain: Optional[str] = "trading"  # trading, technical, general


class AITranslationService:
    """
    AI-powered translation service with Claude integration
    Provides real-time translation for UI components
    """

    def __init__(self):
        self.cache = {}
        self.supported_languages = [Language.ENGLISH, Language.RUSSIAN]
        self.translations_path = Path("webapp/i18n")
        self.ai_model = "claude-4-sonnet"

    async def translate(self, request: TranslationRequest) -> str:
        """
        Translate text using AI
        """
        cache_key = f"{request.text}_{request.source_lang.value}_{request.target_lang.value}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # For properties files, use direct mapping
        if request.domain == "ui":
            translation = await self._translate_ui_text(request)
        else:
            translation = await self._translate_with_ai(request)

        self.cache[cache_key] = translation
        return translation

    async def _translate_ui_text(self, request: TranslationRequest) -> str:
        """
        Translate UI text using properties files
        """
        # Load translation mappings
        en_props = self._load_properties("i18n.properties")
        ru_props = self._load_properties("i18n_ru.properties")

        if request.target_lang == Language.RUSSIAN:
            # Find key in English and return Russian value
            for key, value in en_props.items():
                if value == request.text:
                    return ru_props.get(key, request.text)

        return request.text

    async def _translate_with_ai(self, request: TranslationRequest) -> str:
        """
        Use Claude AI for dynamic translation
        """
        prompt = self._build_translation_prompt(request)

        # Simulate AI translation (would integrate with real Claude API)
        translations = {
            "Market Overview": "Обзор рынка",
            "Portfolio Management": "Управление портфелем",
            "Trading Console": "Торговая консоль",
            "AI Intelligence": "AI Аналитика",
            "Buy": "Купить",
            "Sell": "Продать",
            "Trade": "Торговать",
            "Settings": "Настройки",
            "Welcome": "Добро пожаловать",
            "Loading": "Загрузка",
            "Error": "Ошибка",
            "Success": "Успешно",
            "Cancel": "Отмена",
            "Confirm": "Подтвердить",
            "Price": "Цена",
            "Volume": "Объем",
            "24h Change": "Изменение за 24ч",
            "Market Cap": "Рыночная капитализация",
            "Total Value": "Общая стоимость",
            "Available Balance": "Доступный баланс",
            "Order Type": "Тип ордера",
            "Limit Order": "Лимитный ордер",
            "Market Order": "Рыночный ордер",
            "Stop Loss": "Стоп-лосс",
            "Take Profit": "Тейк-профит",
        }

        if request.target_lang == Language.RUSSIAN:
            return translations.get(request.text, request.text)

        return request.text

    def _build_translation_prompt(self, request: TranslationRequest) -> str:
        """
        Build prompt for Claude AI translation
        """
        context_str = f" Context: {request.context}" if request.context else ""
        domain_str = f" Domain: {request.domain}" if request.domain else ""

        return f"""Translate the following text from {request.source_lang.value} to {request.target_lang.value}.
        Text: "{request.text}"
        {context_str}
        {domain_str}
        
        Provide only the translated text without any explanation."""

    def _load_properties(self, filename: str) -> Dict[str, str]:
        """
        Load properties file
        """
        props = {}
        file_path = self.translations_path / filename

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        props[key.strip()] = value.strip()

        return props

    async def batch_translate(
        self, texts: List[str], source_lang: Language, target_lang: Language
    ) -> List[str]:
        """
        Translate multiple texts in batch
        """
        tasks = []
        for text in texts:
            request = TranslationRequest(text, source_lang, target_lang)
            tasks.append(self.translate(request))

        return await asyncio.gather(*tasks)

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """
        Get list of supported languages
        """
        return [
            {"code": "en", "name": "English", "flag": "🇬🇧"},
            {"code": "ru", "name": "Русский", "flag": "🇷🇺"},
        ]

    async def auto_detect_language(self, text: str) -> Language:
        """
        Auto-detect language of text
        """
        # Simple detection based on character set
        if any("\u0400" <= char <= "\u04FF" for char in text):
            return Language.RUSSIAN
        return Language.ENGLISH


class TranslationMiddleware:
    """
    Middleware for automatic UI translation
    """

    def __init__(self, translation_service: AITranslationService):
        self.service = translation_service
        self.user_language = Language.ENGLISH

    def set_user_language(self, lang_code: str):
        """
        Set user's preferred language
        """
        try:
            self.user_language = Language(lang_code)
        except ValueError:
            logger.warning(f"Unsupported language: {lang_code}")
            self.user_language = Language.ENGLISH

    async def translate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically translate API response
        """
        if self.user_language == Language.ENGLISH:
            return response

        # Recursively translate string values
        return await self._translate_dict(response)

    async def _translate_dict(self, data: Any) -> Any:
        """
        Recursively translate dictionary values
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = await self._translate_dict(value)
            return result
        elif isinstance(data, list):
            return [await self._translate_dict(item) for item in data]
        elif isinstance(data, str):
            request = TranslationRequest(
                text=data, source_lang=Language.ENGLISH, target_lang=self.user_language
            )
            return await self.service.translate(request)
        else:
            return data


# Singleton instance
translation_service = AITranslationService()
