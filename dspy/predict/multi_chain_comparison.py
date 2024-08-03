import dspy
from dspy.signatures.signature import ensure_signature

from ..primitives.program import Module
from .predict import Predict


class MultiChainComparison(Module):
    def __init__(self, signature, M=3, temperature=0.7, **config):
        super().__init__()

        self.M = M
        signature = ensure_signature(signature)

        *_, self.last_key = signature.output_fields.keys()

        for idx in range(M):
            signature = signature.append(
                f"reasoning attempt_{idx+1}",
                dspy.InputField(
                    prefix=f"Studentpoging #{idx+1}:", desc="${reasoning attempt}",
                ),
            )

        signature = signature.prepend(
            "redenering",
            dspy.OutputField(
                prefix="Nauwkeurige Redenering: Bedankt iedereen. Laten we nu holistisch",
                desc="${corrected reasoning}",
            ),
        )

        self.predict = Predict(signature, temperature=temperature, **config)

    def forward(self, completions, **kwargs):
        attempts = []

        for c in completions:
            rationale = c.rationale.strip().split("\n")[0].strip()
            answer = c[self.last_key].strip().split("\n")[0].strip()
            attempts.append(
                f"«Ik probeer te{rationale} Ik weet het niet zeker, maar mijn voorspelling is {answer}»",
            )

        assert len(attempts) == self.M, f"Het aantal pogingen ({len(attempts)}) komt niet overeen met het verwachte aantal M ({self.M}). Stel de juiste waarde voor M in bij het initialiseren van MultiChainVergelijking."

        kwargs = {
            **{
                f"reasoning_attempt_{idx+1}": attempt
                for idx, attempt in enumerate(attempts)
            },
            **kwargs,
        }
        return self.predict(**kwargs)
