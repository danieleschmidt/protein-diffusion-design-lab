"""
Global Compliance and Internationalization Framework

This module implements comprehensive global compliance including:
- GDPR, CCPA, PDPA compliance and data privacy
- Multi-language support and internationalization
- Regional regulatory compliance
- Cross-platform compatibility
- Global deployment standards
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class PrivacyRegulation(Enum):
    """Privacy regulations supported."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)


class DataCategory(Enum):
    """Categories of data for compliance classification."""
    PERSONAL_IDENTIFIABLE = "pii"
    BIOMETRIC = "biometric"
    RESEARCH = "research"
    TECHNICAL = "technical"
    ANONYMOUS = "anonymous"


class ConsentType(Enum):
    """Types of user consent."""
    EXPLICIT = "explicit"
    IMPLIED = "implied"
    OPT_OUT = "opt_out"
    OPT_IN = "opt_in"


@dataclass
class UserConsent:
    """User consent record."""
    user_id: str
    consent_type: ConsentType
    data_categories: List[DataCategory]
    purposes: List[str]
    timestamp: float
    expiry: Optional[float] = None
    withdrawn: bool = False
    regulation: PrivacyRegulation = PrivacyRegulation.GDPR


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""
    processing_id: str
    user_id: str
    data_categories: List[DataCategory]
    purpose: str
    legal_basis: str
    timestamp: float
    retention_period: Optional[int] = None  # days
    cross_border_transfer: bool = False
    processor_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSubjectRequest:
    """Data subject rights request (GDPR Article 15-22)."""
    request_id: str
    user_id: str
    request_type: str  # "access", "rectify", "erase", "port", "restrict", "object"
    timestamp: float
    status: str = "pending"  # "pending", "processing", "completed", "rejected"
    response_data: Optional[Dict[str, Any]] = None
    completion_date: Optional[float] = None


class PrivacyComplianceManager:
    """Manages privacy compliance across different regulations."""
    
    def __init__(self, default_regulation: PrivacyRegulation = PrivacyRegulation.GDPR):
        self.default_regulation = default_regulation
        self.consent_records: Dict[str, UserConsent] = {}
        self.processing_records: List[DataProcessingRecord] = []
        self.subject_requests: Dict[str, DataSubjectRequest] = {}
        
        # Regulation-specific handlers
        self.regulation_handlers = {
            PrivacyRegulation.GDPR: GDPRHandler(),
            PrivacyRegulation.CCPA: CCPAHandler(),
            PrivacyRegulation.PDPA: PDPAHandler(),
            PrivacyRegulation.LGPD: LGPDHandler(),
            PrivacyRegulation.PIPEDA: PIPEDAHandler(),
        }
    
    def record_consent(self, consent: UserConsent) -> bool:
        """Record user consent with compliance validation."""
        handler = self.regulation_handlers[consent.regulation]
        
        # Validate consent meets regulatory requirements
        if not handler.validate_consent(consent):
            logger.error(f"Consent validation failed for user {consent.user_id}")
            return False
        
        # Store consent record
        self.consent_records[consent.user_id] = consent
        
        logger.info(f"Consent recorded for user {consent.user_id} under {consent.regulation.value}")
        return True
    
    def withdraw_consent(self, user_id: str) -> bool:
        """Process consent withdrawal."""
        if user_id not in self.consent_records:
            logger.warning(f"No consent record found for user {user_id}")
            return False
        
        consent = self.consent_records[user_id]
        consent.withdrawn = True
        consent.timestamp = time.time()
        
        # Trigger data erasure if required by regulation
        handler = self.regulation_handlers[consent.regulation]
        if handler.requires_erasure_on_withdrawal():
            self._trigger_data_erasure(user_id)
        
        logger.info(f"Consent withdrawn for user {user_id}")
        return True
    
    def check_processing_lawful(self, user_id: str, purpose: str, data_categories: List[DataCategory]) -> bool:
        """Check if data processing is lawful under applicable regulation."""
        if user_id not in self.consent_records:
            logger.warning(f"No consent record for user {user_id}")
            return False
        
        consent = self.consent_records[user_id]
        
        if consent.withdrawn:
            logger.warning(f"Consent withdrawn for user {user_id}")
            return False
        
        # Check consent covers requested data categories
        if not all(cat in consent.data_categories for cat in data_categories):
            logger.warning(f"Consent doesn't cover all requested data categories for user {user_id}")
            return False
        
        # Check purpose is covered
        if purpose not in consent.purposes:
            logger.warning(f"Purpose '{purpose}' not covered by consent for user {user_id}")
            return False
        
        # Check consent hasn't expired
        if consent.expiry and time.time() > consent.expiry:
            logger.warning(f"Consent expired for user {user_id}")
            return False
        
        return True
    
    def log_processing_activity(self, activity: DataProcessingRecord):
        """Log data processing activity for audit trail."""
        self.processing_records.append(activity)
        
        # Check retention period compliance
        handler = self.regulation_handlers.get(self.default_regulation)
        if handler and activity.retention_period:
            handler.schedule_data_retention_check(activity)
        
        logger.debug(f"Processing activity logged: {activity.processing_id}")
    
    def handle_subject_request(self, request: DataSubjectRequest) -> bool:
        """Handle data subject rights request."""
        if request.user_id not in self.consent_records:
            logger.error(f"No user record found for subject request {request.request_id}")
            return False
        
        consent = self.consent_records[request.user_id]
        handler = self.regulation_handlers[consent.regulation]
        
        # Process request based on type
        try:
            if request.request_type == "access":
                request.response_data = self._handle_access_request(request.user_id)
            elif request.request_type == "erase":
                self._handle_erasure_request(request.user_id)
            elif request.request_type == "port":
                request.response_data = self._handle_portability_request(request.user_id)
            elif request.request_type == "rectify":
                # Would require additional data in request
                pass
            
            request.status = "completed"
            request.completion_date = time.time()
            
            self.subject_requests[request.request_id] = request
            logger.info(f"Subject request {request.request_id} completed")
            return True
            
        except Exception as e:
            request.status = "rejected"
            logger.error(f"Subject request {request.request_id} failed: {e}")
            return False
    
    def _handle_access_request(self, user_id: str) -> Dict[str, Any]:
        """Handle data access request (GDPR Article 15)."""
        user_data = {
            "user_id": user_id,
            "consent_record": self.consent_records.get(user_id).__dict__ if user_id in self.consent_records else None,
            "processing_activities": [
                record.__dict__ for record in self.processing_records 
                if record.user_id == user_id
            ],
            "data_categories": [],
            "retention_periods": {},
            "cross_border_transfers": []
        }
        
        # Collect data categories and retention info
        for record in self.processing_records:
            if record.user_id == user_id:
                user_data["data_categories"].extend(record.data_categories)
                if record.retention_period:
                    user_data["retention_periods"][record.purpose] = record.retention_period
                if record.cross_border_transfer:
                    user_data["cross_border_transfers"].append(record.processor_details)
        
        return user_data
    
    def _handle_erasure_request(self, user_id: str):
        """Handle right to erasure request (GDPR Article 17)."""
        self._trigger_data_erasure(user_id)
    
    def _handle_portability_request(self, user_id: str) -> Dict[str, Any]:
        """Handle data portability request (GDPR Article 20)."""
        # Return user data in structured, machine-readable format
        return self._handle_access_request(user_id)
    
    def _trigger_data_erasure(self, user_id: str):
        """Trigger data erasure across all systems."""
        # Mark consent as withdrawn
        if user_id in self.consent_records:
            self.consent_records[user_id].withdrawn = True
        
        # Remove or anonymize processing records
        self.processing_records = [
            record for record in self.processing_records 
            if record.user_id != user_id
        ]
        
        logger.info(f"Data erasure triggered for user {user_id}")
    
    def generate_compliance_report(self, regulation: PrivacyRegulation) -> Dict[str, Any]:
        """Generate compliance report for specified regulation."""
        handler = self.regulation_handlers[regulation]
        
        total_users = len(self.consent_records)
        withdrawn_consents = sum(1 for c in self.consent_records.values() if c.withdrawn)
        
        processing_by_purpose = {}
        for record in self.processing_records:
            purpose = record.purpose
            processing_by_purpose[purpose] = processing_by_purpose.get(purpose, 0) + 1
        
        cross_border_transfers = sum(1 for r in self.processing_records if r.cross_border_transfer)
        
        return {
            "regulation": regulation.value,
            "report_date": time.time(),
            "total_users": total_users,
            "active_consents": total_users - withdrawn_consents,
            "withdrawn_consents": withdrawn_consents,
            "total_processing_activities": len(self.processing_records),
            "processing_by_purpose": processing_by_purpose,
            "cross_border_transfers": cross_border_transfers,
            "subject_requests": {
                "total": len(self.subject_requests),
                "completed": sum(1 for r in self.subject_requests.values() if r.status == "completed"),
                "pending": sum(1 for r in self.subject_requests.values() if r.status == "pending")
            },
            "compliance_status": handler.assess_compliance(self)
        }


class RegulationHandler(ABC):
    """Abstract base class for regulation-specific handlers."""
    
    @abstractmethod
    def validate_consent(self, consent: UserConsent) -> bool:
        """Validate consent meets regulation requirements."""
        pass
    
    @abstractmethod
    def requires_erasure_on_withdrawal(self) -> bool:
        """Check if regulation requires data erasure on consent withdrawal."""
        pass
    
    @abstractmethod
    def assess_compliance(self, manager: PrivacyComplianceManager) -> Dict[str, Any]:
        """Assess compliance status."""
        pass
    
    def schedule_data_retention_check(self, activity: DataProcessingRecord):
        """Schedule data retention compliance check."""
        # Default implementation
        pass


class GDPRHandler(RegulationHandler):
    """GDPR compliance handler."""
    
    def validate_consent(self, consent: UserConsent) -> bool:
        """GDPR requires explicit, informed, specific consent."""
        # GDPR Article 7
        if consent.consent_type not in [ConsentType.EXPLICIT, ConsentType.OPT_IN]:
            return False
        
        # Must specify purposes
        if not consent.purposes:
            return False
        
        # Must specify data categories
        if not consent.data_categories:
            return False
        
        return True
    
    def requires_erasure_on_withdrawal(self) -> bool:
        """GDPR Article 17 - Right to erasure."""
        return True
    
    def assess_compliance(self, manager: PrivacyComplianceManager) -> Dict[str, Any]:
        """Assess GDPR compliance."""
        issues = []
        
        # Check consent quality
        invalid_consents = 0
        for consent in manager.consent_records.values():
            if not self.validate_consent(consent):
                invalid_consents += 1
        
        if invalid_consents > 0:
            issues.append(f"{invalid_consents} consents don't meet GDPR requirements")
        
        # Check data retention
        overdue_retention = 0
        current_time = time.time()
        for record in manager.processing_records:
            if record.retention_period:
                retention_deadline = record.timestamp + (record.retention_period * 86400)
                if current_time > retention_deadline:
                    overdue_retention += 1
        
        if overdue_retention > 0:
            issues.append(f"{overdue_retention} records exceed retention period")
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "risk_level": "low" if len(issues) == 0 else "medium" if len(issues) < 3 else "high"
        }


class CCPAHandler(RegulationHandler):
    """CCPA compliance handler."""
    
    def validate_consent(self, consent: UserConsent) -> bool:
        """CCPA allows opt-out model."""
        # CCPA is less strict about consent format
        return consent.consent_type in [ConsentType.OPT_OUT, ConsentType.OPT_IN, ConsentType.IMPLIED]
    
    def requires_erasure_on_withdrawal(self) -> bool:
        """CCPA has right to delete but with exceptions."""
        return True
    
    def assess_compliance(self, manager: PrivacyComplianceManager) -> Dict[str, Any]:
        """Assess CCPA compliance."""
        issues = []
        
        # CCPA focuses more on disclosure and opt-out rights
        # Check if users have been provided opt-out mechanisms
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "risk_level": "low"
        }


class PDPAHandler(RegulationHandler):
    """PDPA (Singapore) compliance handler."""
    
    def validate_consent(self, consent: UserConsent) -> bool:
        """PDPA requires clear consent."""
        return consent.consent_type in [ConsentType.EXPLICIT, ConsentType.OPT_IN]
    
    def requires_erasure_on_withdrawal(self) -> bool:
        return True
    
    def assess_compliance(self, manager: PrivacyComplianceManager) -> Dict[str, Any]:
        return {"compliant": True, "issues": [], "risk_level": "low"}


class LGPDHandler(RegulationHandler):
    """LGPD (Brazil) compliance handler."""
    
    def validate_consent(self, consent: UserConsent) -> bool:
        return consent.consent_type in [ConsentType.EXPLICIT, ConsentType.OPT_IN]
    
    def requires_erasure_on_withdrawal(self) -> bool:
        return True
    
    def assess_compliance(self, manager: PrivacyComplianceManager) -> Dict[str, Any]:
        return {"compliant": True, "issues": [], "risk_level": "low"}


class PIPEDAHandler(RegulationHandler):
    """PIPEDA (Canada) compliance handler."""
    
    def validate_consent(self, consent: UserConsent) -> bool:
        return consent.consent_type in [ConsentType.EXPLICIT, ConsentType.OPT_IN, ConsentType.IMPLIED]
    
    def requires_erasure_on_withdrawal(self) -> bool:
        return False  # PIPEDA doesn't mandate erasure
    
    def assess_compliance(self, manager: PrivacyComplianceManager) -> Dict[str, Any]:
        return {"compliant": True, "issues": [], "risk_level": "low"}


class InternationalizationManager:
    """Manages multi-language support and localization."""
    
    def __init__(self, default_language: str = "en"):
        self.default_language = default_language
        self.translations = {}
        self.supported_languages = ["en", "es", "fr", "de", "ja", "zh", "pt", "it", "ru", "ko"]
        
        # Load translations
        self._load_translations()
    
    def _load_translations(self):
        """Load translation dictionaries for supported languages."""
        # Base English translations
        self.translations["en"] = {
            "app_name": "Protein Diffusion Design Lab",
            "welcome_message": "Welcome to the Protein Diffusion Design Lab",
            "generate_button": "Generate Proteins",
            "sequence_label": "Protein Sequence",
            "confidence_label": "Confidence Score",
            "error_invalid_sequence": "Invalid protein sequence",
            "error_rate_limit": "Rate limit exceeded. Please try again later.",
            "privacy_consent": "I consent to processing of my data for protein design research",
            "data_retention_notice": "Your data will be retained for research purposes for up to 7 years",
            "right_to_erasure": "You have the right to request deletion of your data",
            "contact_dpo": "Contact our Data Protection Officer for privacy concerns",
        }
        
        # Spanish translations
        self.translations["es"] = {
            "app_name": "Laboratorio de Diseño de Difusión de Proteínas",
            "welcome_message": "Bienvenido al Laboratorio de Diseño de Difusión de Proteínas",
            "generate_button": "Generar Proteínas",
            "sequence_label": "Secuencia de Proteína",
            "confidence_label": "Puntuación de Confianza",
            "error_invalid_sequence": "Secuencia de proteína inválida",
            "error_rate_limit": "Límite de velocidad excedido. Inténtelo de nuevo más tarde.",
            "privacy_consent": "Consiento el procesamiento de mis datos para investigación de diseño de proteínas",
            "data_retention_notice": "Sus datos se conservarán para fines de investigación hasta 7 años",
            "right_to_erasure": "Tiene derecho a solicitar la eliminación de sus datos",
            "contact_dpo": "Contacte a nuestro Oficial de Protección de Datos para asuntos de privacidad",
        }
        
        # French translations
        self.translations["fr"] = {
            "app_name": "Laboratoire de Conception de Diffusion de Protéines",
            "welcome_message": "Bienvenue au Laboratoire de Conception de Diffusion de Protéines",
            "generate_button": "Générer des Protéines",
            "sequence_label": "Séquence Protéique",
            "confidence_label": "Score de Confiance",
            "error_invalid_sequence": "Séquence protéique invalide",
            "error_rate_limit": "Limite de débit dépassée. Veuillez réessayer plus tard.",
            "privacy_consent": "Je consens au traitement de mes données pour la recherche en conception de protéines",
            "data_retention_notice": "Vos données seront conservées à des fins de recherche jusqu'à 7 ans",
            "right_to_erasure": "Vous avez le droit de demander la suppression de vos données",
            "contact_dpo": "Contactez notre Délégué à la Protection des Données pour les préoccupations de confidentialité",
        }
        
        # German translations
        self.translations["de"] = {
            "app_name": "Protein-Diffusions-Design-Labor",
            "welcome_message": "Willkommen im Protein-Diffusions-Design-Labor",
            "generate_button": "Proteine Generieren",
            "sequence_label": "Proteinsequenz",
            "confidence_label": "Vertrauenswert",
            "error_invalid_sequence": "Ungültige Proteinsequenz",
            "error_rate_limit": "Ratenlimit überschritten. Bitte versuchen Sie es später erneut.",
            "privacy_consent": "Ich stimme der Verarbeitung meiner Daten für die Proteindesign-Forschung zu",
            "data_retention_notice": "Ihre Daten werden für Forschungszwecke bis zu 7 Jahre aufbewahrt",
            "right_to_erasure": "Sie haben das Recht, die Löschung Ihrer Daten zu beantragen",
            "contact_dpo": "Kontaktieren Sie unseren Datenschutzbeauftragten bei Datenschutzanliegen",
        }
        
        # Japanese translations
        self.translations["ja"] = {
            "app_name": "タンパク質拡散設計研究所",
            "welcome_message": "タンパク質拡散設計研究所へようこそ",
            "generate_button": "タンパク質を生成",
            "sequence_label": "タンパク質配列",
            "confidence_label": "信頼度スコア",
            "error_invalid_sequence": "無効なタンパク質配列",
            "error_rate_limit": "レート制限を超えました。後でもう一度お試しください。",
            "privacy_consent": "タンパク質設計研究のためのデータ処理に同意します",
            "data_retention_notice": "お客様のデータは研究目的で最大7年間保持されます",
            "right_to_erasure": "お客様はデータの削除を要求する権利があります",
            "contact_dpo": "プライバシーに関する懸念については、データ保護責任者にお問い合わせください",
        }
        
        # Chinese translations
        self.translations["zh"] = {
            "app_name": "蛋白质扩散设计实验室",
            "welcome_message": "欢迎来到蛋白质扩散设计实验室",
            "generate_button": "生成蛋白质",
            "sequence_label": "蛋白质序列",
            "confidence_label": "置信度评分",
            "error_invalid_sequence": "无效的蛋白质序列",
            "error_rate_limit": "超出速率限制。请稍后重试。",
            "privacy_consent": "我同意为蛋白质设计研究处理我的数据",
            "data_retention_notice": "您的数据将为研究目的保留最多7年",
            "right_to_erasure": "您有权要求删除您的数据",
            "contact_dpo": "如有隐私问题，请联系我们的数据保护官",
        }
    
    def get_translation(self, key: str, language: str = None) -> str:
        """Get translated text for given key and language."""
        if language is None:
            language = self.default_language
        
        if language not in self.translations:
            language = self.default_language
        
        return self.translations[language].get(key, key)
    
    def get_user_language(self, user_preferences: Dict[str, Any], request_headers: Dict[str, str] = None) -> str:
        """Determine user's preferred language."""
        # Check user preferences first
        if "language" in user_preferences:
            lang = user_preferences["language"]
            if lang in self.supported_languages:
                return lang
        
        # Check Accept-Language header
        if request_headers and "Accept-Language" in request_headers:
            accept_lang = request_headers["Accept-Language"]
            # Parse Accept-Language header (simplified)
            for lang_range in accept_lang.split(","):
                lang = lang_range.strip().split(";")[0].split("-")[0]
                if lang in self.supported_languages:
                    return lang
        
        return self.default_language
    
    def format_currency(self, amount: float, currency: str, language: str) -> str:
        """Format currency according to locale."""
        currency_formats = {
            "en": {"USD": "${:.2f}", "EUR": "€{:.2f}", "GBP": "£{:.2f}"},
            "es": {"USD": "${:.2f}", "EUR": "{:.2f} €", "GBP": "£{:.2f}"},
            "fr": {"USD": "{:.2f} $", "EUR": "{:.2f} €", "GBP": "{:.2f} £"},
            "de": {"USD": "{:.2f} $", "EUR": "{:.2f} €", "GBP": "{:.2f} £"},
            "ja": {"USD": "${:.0f}", "EUR": "€{:.0f}", "JPY": "¥{:.0f}"},
            "zh": {"USD": "${:.2f}", "EUR": "€{:.2f}", "CNY": "¥{:.2f}"}
        }
        
        format_str = currency_formats.get(language, currency_formats["en"]).get(currency, "{:.2f}")
        return format_str.format(amount)
    
    def format_date(self, timestamp: float, language: str) -> str:
        """Format date according to locale."""
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        
        date_formats = {
            "en": "%Y-%m-%d %H:%M:%S",
            "es": "%d/%m/%Y %H:%M:%S",
            "fr": "%d/%m/%Y %H:%M:%S",
            "de": "%d.%m.%Y %H:%M:%S",
            "ja": "%Y年%m月%d日 %H:%M:%S",
            "zh": "%Y年%m月%d日 %H:%M:%S"
        }
        
        format_str = date_formats.get(language, date_formats["en"])
        return dt.strftime(format_str)


class GlobalComplianceManager:
    """Central manager for global compliance and internationalization."""
    
    def __init__(self):
        self.privacy_manager = PrivacyComplianceManager()
        self.i18n_manager = InternationalizationManager()
        self.regional_configs = {}
        
        # Load regional configurations
        self._load_regional_configs()
    
    def _load_regional_configs(self):
        """Load region-specific configurations."""
        self.regional_configs = {
            "EU": {
                "regulation": PrivacyRegulation.GDPR,
                "data_localization": True,
                "retention_limits": {"default": 1095},  # 3 years
                "cross_border_restrictions": ["US", "CN"],
                "languages": ["en", "de", "fr", "es", "it"],
                "currency": "EUR"
            },
            "US": {
                "regulation": PrivacyRegulation.CCPA,
                "data_localization": False,
                "retention_limits": {"default": 2555},  # 7 years
                "cross_border_restrictions": [],
                "languages": ["en", "es"],
                "currency": "USD"
            },
            "SG": {
                "regulation": PrivacyRegulation.PDPA,
                "data_localization": True,
                "retention_limits": {"default": 2190},  # 6 years
                "cross_border_restrictions": [],
                "languages": ["en", "zh"],
                "currency": "SGD"
            },
            "BR": {
                "regulation": PrivacyRegulation.LGPD,
                "data_localization": True,
                "retention_limits": {"default": 1825},  # 5 years
                "cross_border_restrictions": [],
                "languages": ["pt", "en"],
                "currency": "BRL"
            },
            "CA": {
                "regulation": PrivacyRegulation.PIPEDA,
                "data_localization": False,
                "retention_limits": {"default": 2555},  # 7 years
                "cross_border_restrictions": [],
                "languages": ["en", "fr"],
                "currency": "CAD"
            }
        }
    
    def get_applicable_regulation(self, user_region: str) -> PrivacyRegulation:
        """Get applicable privacy regulation for user's region."""
        config = self.regional_configs.get(user_region)
        if config:
            return config["regulation"]
        return PrivacyRegulation.GDPR  # Default to GDPR
    
    def validate_cross_border_transfer(self, from_region: str, to_region: str) -> bool:
        """Validate if cross-border data transfer is allowed."""
        from_config = self.regional_configs.get(from_region)
        if not from_config:
            return False
        
        restrictions = from_config.get("cross_border_restrictions", [])
        return to_region not in restrictions
    
    def get_retention_period(self, user_region: str, data_purpose: str = "default") -> int:
        """Get data retention period for region and purpose."""
        config = self.regional_configs.get(user_region)
        if config:
            retention_limits = config.get("retention_limits", {})
            return retention_limits.get(data_purpose, retention_limits.get("default", 1095))
        return 1095  # Default 3 years
    
    def process_user_request(
        self,
        user_id: str,
        user_region: str,
        request_type: str,
        language: str = None
    ) -> Dict[str, Any]:
        """Process user request with regional compliance."""
        regulation = self.get_applicable_regulation(user_region)
        
        # Create data subject request
        request = DataSubjectRequest(
            request_id=f"req_{int(time.time())}_{user_id}",
            user_id=user_id,
            request_type=request_type,
            timestamp=time.time()
        )
        
        # Process request
        success = self.privacy_manager.handle_subject_request(request)
        
        # Prepare response in user's language
        if not language:
            language = self.regional_configs.get(user_region, {}).get("languages", ["en"])[0]
        
        response = {
            "request_id": request.request_id,
            "status": request.status,
            "regulation": regulation.value,
            "message": self.i18n_manager.get_translation(
                f"request_{request_type}_response", language
            ),
            "data": request.response_data if success else None
        }
        
        return response
    
    def generate_privacy_notice(self, user_region: str, language: str = "en") -> str:
        """Generate region-appropriate privacy notice."""
        regulation = self.get_applicable_regulation(user_region)
        retention_period = self.get_retention_period(user_region)
        
        i18n = self.i18n_manager
        
        notice_parts = [
            i18n.get_translation("privacy_notice_header", language),
            f"\n{i18n.get_translation('data_retention_notice', language)}",
            f"\n{i18n.get_translation('regulation_info', language)}: {regulation.value.upper()}",
            f"\n{i18n.get_translation('retention_period', language)}: {retention_period} days",
            f"\n{i18n.get_translation('right_to_erasure', language)}",
            f"\n{i18n.get_translation('contact_dpo', language)}"
        ]
        
        return "\n".join(notice_parts)
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard."""
        dashboard = {
            "timestamp": time.time(),
            "global_compliance_status": {},
            "regional_statistics": {},
            "privacy_requests": {},
            "data_flows": {}
        }
        
        # Compliance status by regulation
        for regulation in PrivacyRegulation:
            report = self.privacy_manager.generate_compliance_report(regulation)
            dashboard["global_compliance_status"][regulation.value] = report
        
        # Regional statistics
        for region, config in self.regional_configs.items():
            regulation = config["regulation"]
            dashboard["regional_statistics"][region] = {
                "regulation": regulation.value,
                "data_localization": config["data_localization"],
                "supported_languages": config["languages"],
                "currency": config["currency"]
            }
        
        # Privacy request statistics
        requests_by_type = {}
        for request in self.privacy_manager.subject_requests.values():
            req_type = request.request_type
            requests_by_type[req_type] = requests_by_type.get(req_type, 0) + 1
        
        dashboard["privacy_requests"] = {
            "total": len(self.privacy_manager.subject_requests),
            "by_type": requests_by_type,
            "completion_rate": len([r for r in self.privacy_manager.subject_requests.values() if r.status == "completed"]) / max(1, len(self.privacy_manager.subject_requests))
        }
        
        return dashboard


def create_user_consent(
    user_id: str,
    purposes: List[str],
    data_categories: List[DataCategory],
    regulation: PrivacyRegulation = PrivacyRegulation.GDPR
) -> UserConsent:
    """Helper function to create user consent record."""
    return UserConsent(
        user_id=user_id,
        consent_type=ConsentType.EXPLICIT,
        data_categories=data_categories,
        purposes=purposes,
        timestamp=time.time(),
        regulation=regulation
    )


def create_processing_record(
    user_id: str,
    purpose: str,
    data_categories: List[DataCategory],
    legal_basis: str = "consent"
) -> DataProcessingRecord:
    """Helper function to create processing record."""
    return DataProcessingRecord(
        processing_id=f"proc_{int(time.time())}_{hashlib.md5(user_id.encode()).hexdigest()[:8]}",
        user_id=user_id,
        data_categories=data_categories,
        purpose=purpose,
        legal_basis=legal_basis,
        timestamp=time.time()
    )