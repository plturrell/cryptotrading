"""
SAP CAP SCIP Indexer - Generates Glean-compatible facts for CAP models
Indexes .cds files, service definitions, entities, and associations
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CAPEntity:
    """CAP entity representation"""
    name: str
    namespace: str = ""
    elements: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)
    associations: List[str] = field(default_factory=list)
    compositions: List[str] = field(default_factory=list)

@dataclass
class CAPService:
    """CAP service representation"""
    name: str
    namespace: str = ""
    entities: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CAPDocument:
    """CAP document with parsed content"""
    relative_path: str
    language: str = "cds"
    entities: List[CAPEntity] = field(default_factory=list)
    services: List[CAPService] = field(default_factory=list)
    types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    using_statements: List[str] = field(default_factory=list)

class SAPCAPIndexer:
    """Indexes SAP CAP .cds files into SCIP format for Glean"""
    
    # CDS parsing patterns
    NAMESPACE_PATTERN = re.compile(r'namespace\s+([\w\.]+)\s*;')
    USING_PATTERN = re.compile(r'using\s+([\w\.]+)(?:\s+as\s+(\w+))?\s*;')
    ENTITY_PATTERN = re.compile(r'entity\s+(\w+)\s*(?::\s*\w+\s*)?{([^}]+)}', re.MULTILINE | re.DOTALL)
    SERVICE_PATTERN = re.compile(r'service\s+(\w+)\s*{([^}]+)}', re.MULTILINE | re.DOTALL)
    TYPE_PATTERN = re.compile(r'type\s+(\w+)\s*:\s*([^;]+);')
    ELEMENT_PATTERN = re.compile(r'(\w+)\s*:\s*([^;,}]+)[;,]?')
    ASSOCIATION_PATTERN = re.compile(r'(\w+)\s*:\s*Association\s+to\s+([\w\.]+)')
    COMPOSITION_PATTERN = re.compile(r'(\w+)\s*:\s*Composition\s+of\s+([\w\.]+)')
    ANNOTATION_PATTERN = re.compile(r'@([\w\.]+)(?:\(([^)]+)\))?')
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.documents: List[CAPDocument] = []
        
    def index_file(self, file_path: Path) -> Optional[CAPDocument]:
        """Index a single .cds file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            relative_path = file_path.relative_to(self.project_root)
            document = CAPDocument(relative_path=str(relative_path))
            
            # Parse namespace
            namespace_match = self.NAMESPACE_PATTERN.search(content)
            current_namespace = namespace_match.group(1) if namespace_match else ""
            
            # Parse using statements
            for using_match in self.USING_PATTERN.finditer(content):
                using_stmt = using_match.group(1)
                alias = using_match.group(2)
                document.using_statements.append(f"{using_stmt}{' as ' + alias if alias else ''}")
            
            # Parse entities
            for entity_match in self.ENTITY_PATTERN.finditer(content):
                entity_name = entity_match.group(1)
                entity_body = entity_match.group(2)
                
                entity = CAPEntity(
                    name=entity_name,
                    namespace=current_namespace
                )
                
                # Parse entity elements
                self._parse_entity_elements(entity_body, entity)
                document.entities.append(entity)
            
            # Parse services
            for service_match in self.SERVICE_PATTERN.finditer(content):
                service_name = service_match.group(1)
                service_body = service_match.group(2)
                
                service = CAPService(
                    name=service_name,
                    namespace=current_namespace
                )
                
                # Parse service content
                self._parse_service_content(service_body, service)
                document.services.append(service)
            
            # Parse types
            for type_match in self.TYPE_PATTERN.finditer(content):
                type_name = type_match.group(1)
                type_def = type_match.group(2).strip()
                document.types[type_name] = {"definition": type_def}
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to index CAP file {file_path}: {e}")
            return None
    
    def _parse_entity_elements(self, body: str, entity: CAPEntity):
        """Parse entity elements from body"""
        # Parse associations
        for assoc_match in self.ASSOCIATION_PATTERN.finditer(body):
            element_name = assoc_match.group(1)
            target_entity = assoc_match.group(2)
            entity.associations.append(target_entity)
            entity.elements[element_name] = {
                "type": "Association",
                "target": target_entity
            }
        
        # Parse compositions
        for comp_match in self.COMPOSITION_PATTERN.finditer(body):
            element_name = comp_match.group(1)
            target_entity = comp_match.group(2)
            entity.compositions.append(target_entity)
            entity.elements[element_name] = {
                "type": "Composition",
                "target": target_entity
            }
        
        # Parse regular elements
        for elem_match in self.ELEMENT_PATTERN.finditer(body):
            element_name = elem_match.group(1)
            element_type = elem_match.group(2).strip()
            
            # Skip if already parsed as association/composition
            if element_name not in entity.elements:
                entity.elements[element_name] = {
                    "type": element_type
                }
        
        # Parse annotations
        for anno_match in self.ANNOTATION_PATTERN.finditer(body):
            annotation = anno_match.group(1)
            value = anno_match.group(2) if anno_match.group(2) else True
            entity.annotations[annotation] = value
    
    def _parse_service_content(self, body: str, service: CAPService):
        """Parse service content from body"""
        # Find exposed entities
        entity_refs = re.findall(r'entity\s+(\w+)', body)
        service.entities.extend(entity_refs)
        
        # Find actions and functions
        action_refs = re.findall(r'action\s+(\w+)', body)
        service.actions.extend(action_refs)
        
        function_refs = re.findall(r'function\s+(\w+)', body)
        service.functions.extend(function_refs)
        
        # Parse annotations
        for anno_match in self.ANNOTATION_PATTERN.finditer(body):
            annotation = anno_match.group(1)
            value = anno_match.group(2) if anno_match.group(2) else True
            service.annotations[annotation] = value
    
    def index_directory(self, directory: Path) -> None:
        """Index all .cds files in a directory"""
        for cds_file in directory.rglob("*.cds"):
            doc = self.index_file(cds_file)
            if doc:
                self.documents.append(doc)
                logger.info(f"Indexed CAP file {cds_file}")
    
    def generate_glean_facts(self) -> List[Dict[str, Any]]:
        """Generate Glean facts from CAP documents"""
        facts = []
        
        for doc in self.documents:
            # Generate file facts
            facts.append({
                "predicate": "src.File",
                "key": {"path": doc.relative_path},
                "value": {"language": "CAP"}
            })
            
            # Generate entity facts
            for entity in doc.entities:
                facts.append({
                    "predicate": "cap.Entity",
                    "key": {
                        "name": entity.name,
                        "namespace": entity.namespace,
                        "file": doc.relative_path
                    },
                    "value": {
                        "elements": list(entity.elements.keys()),
                        "associations": entity.associations,
                        "compositions": entity.compositions,
                        "annotations": entity.annotations
                    }
                })
                
                # Generate element facts
                for elem_name, elem_info in entity.elements.items():
                    facts.append({
                        "predicate": "cap.Element",
                        "key": {
                            "entity": entity.name,
                            "name": elem_name,
                            "file": doc.relative_path
                        },
                        "value": {
                            "type": elem_info.get("type", "unknown"),
                            "target": elem_info.get("target")
                        }
                    })
            
            # Generate service facts
            for service in doc.services:
                facts.append({
                    "predicate": "cap.Service",
                    "key": {
                        "name": service.name,
                        "namespace": service.namespace,
                        "file": doc.relative_path
                    },
                    "value": {
                        "entities": service.entities,
                        "actions": service.actions,
                        "functions": service.functions,
                        "annotations": service.annotations
                    }
                })
            
            # Generate type facts
            for type_name, type_info in doc.types.items():
                facts.append({
                    "predicate": "cap.Type",
                    "key": {
                        "name": type_name,
                        "file": doc.relative_path
                    },
                    "value": {
                        "definition": type_info["definition"]
                    }
                })
            
            # Generate using/import facts
            for using_stmt in doc.using_statements:
                facts.append({
                    "predicate": "cap.Using",
                    "key": {
                        "file": doc.relative_path,
                        "import": using_stmt
                    },
                    "value": {}
                })
        
        return facts

def index_cap_project_for_glean(project_path: str) -> Dict[str, Any]:
    """Main entry point to index SAP CAP project for Glean"""
    project_root = Path(project_path)
    indexer = SAPCAPIndexer(project_root)
    
    # Index the entire project for .cds files
    indexer.index_directory(project_root)
    
    # Generate outputs
    return {
        "cap_documents": len(indexer.documents),
        "glean_facts": indexer.generate_glean_facts(),
        "stats": {
            "files_indexed": len(indexer.documents),
            "total_entities": sum(len(doc.entities) for doc in indexer.documents),
            "total_services": sum(len(doc.services) for doc in indexer.documents),
            "total_types": sum(len(doc.types) for doc in indexer.documents)
        }
    }
