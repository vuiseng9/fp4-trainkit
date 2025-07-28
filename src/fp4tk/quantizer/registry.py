from typing import Type

class QuantizerRegistry:
    _registry: dict[str, Type] = {}

    @classmethod
    def register(cls, key: str):
        def decorator(qcls: Type):
            cls._registry[key] = qcls
            return qcls
        return decorator

    @classmethod
    def create(cls, qcfg):
        """
        Look up qformat in the registry
        and instantiate quantizer with cfg.
        """
        try:
            QConstructor = cls._registry[qcfg.quant_format.value]
        except KeyError:
            raise ValueError(f"Unknown quantizer type {qcfg.quant_format.value!r}")
        return QConstructor(qcfg)