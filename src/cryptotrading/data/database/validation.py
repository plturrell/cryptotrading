"""
Database Data Validation and Constraints
Provides data validation, constraint enforcement, and data quality monitoring
"""

import json
import logging
import re
from datetime import datetime
from decimal import Decimal
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when data validation fails"""

    pass


class DataValidator:
    """Validates data before database operations"""

    def __init__(self):
        self.validators: Dict[str, List[Callable]] = {}
        self.custom_validators: Dict[str, Callable] = {}
        self._register_default_validators()

    def _register_default_validators(self):
        """Register default validators for common data types"""
        # Email validation
        self.register_custom_validator("email", self._validate_email)

        # URL validation
        self.register_custom_validator("url", self._validate_url)

        # Crypto address validation
        self.register_custom_validator("crypto_address", self._validate_crypto_address)

        # Price validation
        self.register_custom_validator("price", self._validate_price)

        # Trading symbol validation
        self.register_custom_validator("trading_symbol", self._validate_trading_symbol)

    def register_custom_validator(self, name: str, validator: Callable):
        """Register a custom validator"""
        self.custom_validators[name] = validator

    def register_field_validator(self, table: str, field: str, validator: Callable):
        """Register validator for specific table field"""
        key = f"{table}.{field}"
        if key not in self.validators:
            self.validators[key] = []
        self.validators[key].append(validator)

    def validate(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data for a table"""
        errors = []
        validated_data = data.copy()

        for field, value in data.items():
            key = f"{table}.{field}"

            # Run field-specific validators
            if key in self.validators:
                for validator in self.validators[key]:
                    try:
                        validated_value = validator(value)
                        if validated_value is not None:
                            validated_data[field] = validated_value
                    except ValidationError as e:
                        errors.append(f"{field}: {str(e)}")
                    except Exception as e:
                        errors.append(f"{field}: Validation error - {str(e)}")

        if errors:
            raise ValidationError(f"Validation failed: {'; '.join(errors)}")

        return validated_data

    def _validate_email(self, email: str) -> str:
        """Validate email format"""
        if not email:
            raise ValidationError("Email is required")

        email = email.strip().lower()
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        if not re.match(pattern, email):
            raise ValidationError("Invalid email format")

        return email

    def _validate_url(self, url: str) -> str:
        """Validate URL format"""
        if not url:
            return url

        pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        if not re.match(pattern, url, re.IGNORECASE):
            raise ValidationError("Invalid URL format")

        return url

    def _validate_crypto_address(self, address: str) -> str:
        """Validate cryptocurrency address"""
        if not address:
            raise ValidationError("Address is required")

        # Basic validation - can be extended for specific cryptocurrencies
        if len(address) < 26 or len(address) > 62:
            raise ValidationError("Invalid address length")

        if not re.match(r"^[a-zA-Z0-9]+$", address):
            raise ValidationError("Invalid address format")

        return address

    def _validate_price(self, price: Any) -> Decimal:
        """Validate and convert price"""
        try:
            price_decimal = Decimal(str(price))

            if price_decimal < 0:
                raise ValidationError("Price cannot be negative")

            if price_decimal > Decimal("1000000000"):  # 1 billion
                raise ValidationError("Price exceeds maximum allowed value")

            return price_decimal

        except (ValueError, TypeError):
            raise ValidationError("Invalid price format")

    def _validate_trading_symbol(self, symbol: str) -> str:
        """Validate trading symbol"""
        if not symbol:
            raise ValidationError("Symbol is required")

        symbol = symbol.upper().strip()

        # Common crypto symbol pattern
        if not re.match(r"^[A-Z0-9]{2,10}(-[A-Z]{3,4})?$", symbol):
            raise ValidationError("Invalid trading symbol format")

        return symbol


class ConstraintEnforcer:
    """Enforces database constraints and business rules"""

    def __init__(self, db_client):
        self.db_client = db_client
        self.constraints: Dict[str, List[Callable]] = {}

    def register_constraint(self, table: str, constraint: Callable):
        """Register a constraint for a table"""
        if table not in self.constraints:
            self.constraints[table] = []
        self.constraints[table].append(constraint)

    def check_constraints(
        self, table: str, data: Dict[str, Any], operation: str = "insert"
    ) -> List[str]:
        """Check all constraints for a table"""
        violations = []

        if table not in self.constraints:
            return violations

        for constraint in self.constraints[table]:
            try:
                result = constraint(data, operation, self.db_client)
                if result is False:
                    violations.append(f"Constraint violation in {constraint.__name__}")
                elif isinstance(result, str):
                    violations.append(result)
            except Exception as e:
                violations.append(f"Constraint check failed: {str(e)}")

        return violations

    def enforce_referential_integrity(
        self, table: str, foreign_keys: Dict[str, str], data: Dict[str, Any]
    ) -> List[str]:
        """Enforce foreign key constraints"""
        violations = []

        for fk_field, fk_table_field in foreign_keys.items():
            if fk_field not in data or data[fk_field] is None:
                continue

            fk_table, fk_field_name = fk_table_field.split(".")

            # Check if referenced record exists
            check_query = f"SELECT 1 FROM {fk_table} WHERE {fk_field_name} = ?"

            try:
                with self.db_client.engine.connect() as conn:
                    if self.db_client.is_postgres:
                        check_query = check_query.replace("?", "%s")

                    result = conn.execute(check_query, (data[fk_field],))
                    if not result.fetchone():
                        violations.append(
                            f"Foreign key violation: {fk_field} references "
                            f"non-existent {fk_table}.{fk_field_name}"
                        )
            except Exception as e:
                violations.append(f"Failed to check foreign key {fk_field}: {str(e)}")

        return violations


class DataQualityMonitor:
    """Monitors and reports on data quality"""

    def __init__(self, db_client):
        self.db_client = db_client
        self.quality_checks: Dict[str, Callable] = {}

    def register_quality_check(self, name: str, check: Callable):
        """Register a data quality check"""
        self.quality_checks[name] = check

    def run_quality_checks(self) -> Dict[str, Any]:
        """Run all data quality checks"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": 100.0,
            "checks": {},
            "issues": [],
        }

        total_weight = 0
        weighted_score = 0

        for check_name, check_func in self.quality_checks.items():
            try:
                check_result = check_func(self.db_client)

                results["checks"][check_name] = {
                    "score": check_result.get("score", 100),
                    "weight": check_result.get("weight", 1),
                    "issues": check_result.get("issues", []),
                    "metrics": check_result.get("metrics", {}),
                }

                # Update weighted score
                weight = check_result.get("weight", 1)
                score = check_result.get("score", 100)
                total_weight += weight
                weighted_score += score * weight

                # Collect issues
                if check_result.get("issues"):
                    results["issues"].extend(
                        [{"check": check_name, "issue": issue} for issue in check_result["issues"]]
                    )

            except Exception as e:
                logger.error(f"Quality check '{check_name}' failed: {e}")
                results["checks"][check_name] = {"score": 0, "weight": 1, "error": str(e)}
                results["issues"].append({"check": check_name, "issue": f"Check failed: {str(e)}"})

        # Calculate overall score
        if total_weight > 0:
            results["overall_score"] = weighted_score / total_weight

        # Log results
        self._log_quality_results(results)

        return results

    def check_data_completeness(self, table: str, required_fields: List[str]) -> Dict[str, Any]:
        """Check data completeness for a table"""
        try:
            total_query = f"SELECT COUNT(*) as total FROM {table}"

            with self.db_client.engine.connect() as conn:
                total_count = conn.execute(total_query).fetchone()["total"]

                if total_count == 0:
                    return {"score": 100, "weight": 1, "metrics": {"total_records": 0}}

                issues = []
                null_counts = {}

                for field in required_fields:
                    null_query = f"SELECT COUNT(*) as nulls FROM {table} WHERE {field} IS NULL"
                    null_count = conn.execute(null_query).fetchone()["nulls"]
                    null_counts[field] = null_count

                    if null_count > 0:
                        null_percent = (null_count / total_count) * 100
                        issues.append(f"{field}: {null_count} null values ({null_percent:.1f}%)")

                # Calculate completeness score
                total_nulls = sum(null_counts.values())
                total_possible = total_count * len(required_fields)
                completeness = ((total_possible - total_nulls) / total_possible) * 100

                return {
                    "score": completeness,
                    "weight": 2,  # Higher weight for completeness
                    "issues": issues,
                    "metrics": {
                        "total_records": total_count,
                        "null_counts": null_counts,
                        "completeness_percent": completeness,
                    },
                }

        except Exception as e:
            logger.error(f"Completeness check failed for {table}: {e}")
            return {"score": 0, "weight": 2, "error": str(e)}

    def check_data_consistency(self) -> Dict[str, Any]:
        """Check data consistency across tables"""
        issues = []
        metrics = {}

        try:
            # Check trade-portfolio consistency
            inconsistent_trades = self._check_trade_portfolio_consistency()
            if inconsistent_trades:
                issues.append(f"{len(inconsistent_trades)} trades not reflected in portfolios")
                metrics["inconsistent_trades"] = inconsistent_trades

            # Check orphaned records
            orphans = self._check_orphaned_records()
            for table, count in orphans.items():
                if count > 0:
                    issues.append(f"{count} orphaned records in {table}")
            metrics["orphaned_records"] = orphans

            # Calculate consistency score
            total_issues = len(inconsistent_trades) + sum(orphans.values())
            score = max(0, 100 - (total_issues * 5))  # Deduct 5 points per issue

            return {
                "score": score,
                "weight": 3,  # High weight for consistency
                "issues": issues,
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return {"score": 0, "weight": 3, "error": str(e)}

    def _check_trade_portfolio_consistency(self) -> List[int]:
        """Check if all trades are properly reflected in portfolios"""
        query = """
        SELECT t.id
        FROM trades t
        LEFT JOIN portfolios p ON t.user_id = p.user_id AND t.symbol = p.symbol
        WHERE t.status = 'completed' AND p.id IS NULL
        """

        try:
            with self.db_client.engine.connect() as conn:
                result = conn.execute(query)
                return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to get table names: {e}")
            return []

    def _check_orphaned_records(self) -> Dict[str, int]:
        """Check for orphaned records"""
        orphans = {}

        checks = [
            ("trades", "SELECT COUNT(*) FROM trades WHERE user_id NOT IN (SELECT id FROM users)"),
            (
                "portfolios",
                "SELECT COUNT(*) FROM portfolios WHERE user_id NOT IN (SELECT id FROM users)",
            ),
            (
                "conversation_messages",
                "SELECT COUNT(*) FROM conversation_messages WHERE session_id NOT IN (SELECT session_id FROM conversation_sessions)",
            ),
        ]

        try:
            with self.db_client.engine.connect() as conn:
                for table, query in checks:
                    try:
                        result = conn.execute(query)
                        orphans[table] = result.fetchone()[0]
                    except Exception as e:
                        logger.warning(f"Failed to check orphaned records for {table}: {e}")
                        orphans[table] = -1  # Indicate check failure
        except Exception as e:
            logger.error(f"Database connection failed during orphan check: {e}")
            # Return partial results if any tables were checked
            for table, _ in checks:
                if table not in orphans:
                    orphans[table] = -1

        return orphans

    def _log_quality_results(self, results: Dict[str, Any]):
        """Log quality check results"""
        try:
            for check_name, check_result in results["checks"].items():
                if "error" not in check_result:
                    insert_sql = """
                    INSERT INTO data_quality_metrics
                    (table_name, metric_name, metric_value, threshold_value, is_passing)
                    VALUES (?, ?, ?, ?, ?)
                    """

                    is_passing = check_result["score"] >= 80  # 80% threshold

                    with self.db_client.engine.connect() as conn:
                        if self.db_client.is_postgres:
                            insert_sql = insert_sql.replace("?", "%s")

                        conn.execute(
                            insert_sql,
                            ("overall", check_name, check_result["score"], 80.0, is_passing),
                        )
                        conn.commit()

        except Exception as e:
            logger.error(f"Failed to log quality results: {e}")


def validate_model(model_class: Type):
    """Decorator to add validation to SQLAlchemy models"""

    def decorator(cls):
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, **kwargs):
            # Validate data before initialization
            validator = DataValidator()
            validated_data = validator.validate(cls.__tablename__, kwargs)

            # Call original init with validated data
            original_init(self, **validated_data)

        cls.__init__ = new_init
        return cls

    return decorator


# Predefined constraint functions
def unique_constraint(field: str):
    """Create a unique constraint checker"""

    def check(data: Dict[str, Any], operation: str, db_client) -> Optional[str]:
        if operation == "update" or field not in data:
            return None

        table = data.get("__table__", "unknown")
        value = data[field]

        query = f"SELECT 1 FROM {table} WHERE {field} = ? LIMIT 1"

        try:
            with db_client.engine.connect() as conn:
                if db_client.is_postgres:
                    query = query.replace("?", "%s")

                result = conn.execute(query, (value,))
                if result.fetchone():
                    return f"{field} '{value}' already exists"
        except Exception as e:
            logger.error(f"Failed to check uniqueness for {field}='{value}': {e}")
            # Return None to allow operation - constraint will be caught at DB level
            return None

        return None

    return check


def range_constraint(field: str, min_value: Any = None, max_value: Any = None):
    """Create a range constraint checker"""

    def check(data: Dict[str, Any], operation: str, db_client) -> Optional[str]:
        if field not in data:
            return None

        value = data[field]

        if min_value is not None and value < min_value:
            return f"{field} must be >= {min_value}"

        if max_value is not None and value > max_value:
            return f"{field} must be <= {max_value}"

        return None

    return check
