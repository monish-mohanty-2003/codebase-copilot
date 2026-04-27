"""Tree-sitter parsing.

Strategy: use language-specific node-type allowlists to extract definitions
(functions, classes, methods). This is more robust than queries for our
purposes and avoids shipping .scm files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node
from tree_sitter_languages import get_parser

logger = logging.getLogger(__name__)

# File extension -> tree-sitter language name
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

# Node types that represent a "definition" worth chunking, per language.
# Tree-sitter's grammar names — verified against the official grammars.
DEFINITION_NODES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {
        "function_declaration", "class_declaration", "method_definition",
        "arrow_function", "generator_function_declaration",
    },
    "typescript": {
        "function_declaration", "class_declaration", "method_definition",
        "interface_declaration", "type_alias_declaration",
    },
    "tsx": {
        "function_declaration", "class_declaration", "method_definition",
        "interface_declaration", "type_alias_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
}

CLASS_LIKE: set[str] = {
    "class_definition", "class_declaration", "impl_item", "struct_item",
    "interface_declaration", "type_declaration",
}


@dataclass
class ParsedDefinition:
    """A raw extracted definition from the AST. Pre-chunking."""
    kind: str                # 'function' | 'class' | 'method'
    symbol: str              # e.g. 'JWTAuth.verify_token'
    parent_symbol: str | None
    start_line: int          # 1-indexed inclusive
    end_line: int            # 1-indexed inclusive
    content: str
    node_type: str


class TreeSitterParser:
    """Parses a source file and yields top-level definitions."""

    def __init__(self) -> None:
        self._parsers: dict[str, object] = {}

    def _get_parser(self, language: str):
        if language not in self._parsers:
            self._parsers[language] = get_parser(language)
        return self._parsers[language]

    def language_for(self, path: Path) -> str | None:
        return EXTENSION_TO_LANGUAGE.get(path.suffix.lower())

    def parse_file(self, path: Path) -> tuple[str, list[ParsedDefinition], list[str]]:
        """Returns (language, definitions, imports).

        `imports` is a flat list of imported module names (best-effort).
        """
        language = self.language_for(path)
        if language is None:
            return "", [], []

        try:
            source = path.read_bytes()
        except OSError as e:
            logger.warning("Cannot read %s: %s", path, e)
            return language, [], []

        parser = self._get_parser(language)
        tree = parser.parse(source)
        root = tree.root_node

        defs: list[ParsedDefinition] = []
        self._extract_definitions(root, source, language, parent=None, out=defs)

        imports = self._extract_imports(root, source, language)
        return language, defs, imports

    def _extract_definitions(
        self,
        node: Node,
        source: bytes,
        language: str,
        parent: str | None,
        out: list[ParsedDefinition],
    ) -> None:
        """Recursively walk AST collecting definitions. Methods get parent class as prefix."""
        defn_types = DEFINITION_NODES.get(language, set())

        if node.type in defn_types:
            symbol = self._extract_symbol_name(node, source, language)
            if symbol:
                full_symbol = f"{parent}.{symbol}" if parent else symbol
                kind = self._classify_kind(node.type, parent is not None)
                content = source[node.start_byte:node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                out.append(ParsedDefinition(
                    kind=kind,
                    symbol=full_symbol,
                    parent_symbol=parent,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    content=content,
                    node_type=node.type,
                ))
                # Recurse into class-like nodes to capture methods,
                # but skip recursing into function bodies (no nested defs at top level).
                if node.type in CLASS_LIKE:
                    for child in node.children:
                        self._extract_definitions(
                            child, source, language, parent=full_symbol, out=out
                        )
                return  # don't double-process children of a function

        # Not a definition — keep walking
        for child in node.children:
            self._extract_definitions(child, source, language, parent, out)

    @staticmethod
    def _extract_symbol_name(node: Node, source: bytes, language: str) -> str | None:
        """Find the identifier child. Tree-sitter exposes 'name' field for most defns."""
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return source[name_node.start_byte:name_node.end_byte].decode(
                "utf-8", errors="replace"
            )
        # Fallback for languages where 'name' field isn't set on every node
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "property_identifier"):
                return source[child.start_byte:child.end_byte].decode(
                    "utf-8", errors="replace"
                )
        return None

    @staticmethod
    def _classify_kind(node_type: str, has_parent: bool) -> str:
        if node_type in CLASS_LIKE:
            return "class"
        if "method" in node_type or has_parent:
            return "method"
        return "function"

    def _extract_imports(self, root: Node, source: bytes, language: str) -> list[str]:
        """Best-effort import extraction. Used as metadata for filtering."""
        imports: list[str] = []
        import_node_types = {
            "python": {"import_statement", "import_from_statement"},
            "javascript": {"import_statement"},
            "typescript": {"import_statement"},
            "tsx": {"import_statement"},
            "go": {"import_declaration"},
            "rust": {"use_declaration"},
            "java": {"import_declaration"},
        }.get(language, set())

        def walk(n: Node) -> None:
            if n.type in import_node_types:
                text = source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
                imports.append(text.strip())
            for child in n.children:
                walk(child)

        walk(root)
        return imports