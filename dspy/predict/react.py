import dsp
import dspy
from dspy.signatures.signature import ensure_signature

from ..primitives.program import Module
from .predict import Predict

# TODO: Simplify a lot.
# TODO: Divide Action and Action Input like langchain does for ReAct.

# TODO: There's a lot of value in having a stopping condition in the LM calls at `\n\nObservation:`


class ReAct(Module):
    def __init__(self, signature, max_iters=5, num_results=3, tools=None):
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        self.tools = tools or [dspy.Retrieve(k=num_results)]
        self.tools = {tool.name: tool for tool in self.tools}

        self.input_fields = self.signature.input_fields
        self.output_fields = self.signature.output_fields

        assert len(self.output_fields) == 1, "ReAct ondersteunt slechts één uitvoerveld."

        inputs_ = ", ".join([f"`{k}`" for k in self.input_fields.keys()])
        outputs_ = ", ".join([f"`{k}`" for k in self.output_fields.keys()])

        instr = []
        
        if self.signature.instructions is not None:
            instr.append(f"{self.signature.instructions}\n")
        
        instr.extend([
                    f"Je krijgt {inputs_} en je reageert met {outputs_}.\n",
                    "Om dit te doen, wissel je Gedachte-, Actie- en Observatiestappen af.\n",
                    "Gedachte kan redeneren over de huidige situatie, en Actie kan van de volgende types zijn:\n",
                ])

        self.tools["Finish"] = dspy.Example(
            name="Finish",
            input_variable=outputs_.strip("`"),
            desc=f"geeft het uiteindelijke {outputs_} terug en beëindigt de taak",
        )

        for idx, tool in enumerate(self.tools):
            tool = self.tools[tool]
            instr.append(
                f"({idx+1}) {tool.name}[{tool.input_variable}], which {tool.desc}",
            )

        instr = "\n".join(instr)
        self.react = [
            Predict(dspy.Signature(self._generate_signature(i), instr))
            for i in range(1, max_iters + 1)
        ]

    def _generate_signature(self, iters):
        signature_dict = {}
        for key, val in self.input_fields.items():
            signature_dict[key] = val

        for j in range(1, iters + 1):
            IOField = dspy.OutputField if j == iters else dspy.InputField

            signature_dict[f"Thought_{j}"] = IOField(
                prefix=f"Thought {j}:",
                desc="volgende stappen op basis van de laatste observatie",
            )

            tool_list = " or ".join(
                [
                    f"{tool.name}[{tool.input_variable}]"
                    for tool in self.tools.values()
                    if tool.name != "Finish"
                ],
            )
            signature_dict[f"Action_{j}"] = IOField(
                prefix=f"Action {j}:",
                desc=f"altijd ofwel {tool_list} of, wanneer klaar, Finish[<awnser>], waarbij <awnser> het antwoord op de vraag zelf is.",
            )

            if j < iters:
                signature_dict[f"Observation_{j}"] = IOField(
                    prefix=f"Observatie {j}:",
                    desc="observaties gebaseerd op actie",
                    format=dsp.passages2text,
                )

        return signature_dict

    def act(self, output, hop):
        try:
            action = output[f"Action_{hop+1}"]
            action_name, action_val = action.strip().split("\n")[0].split("[", 1)
            action_val = action_val.rsplit("]", 1)[0]

            if action_name == "Finish":
                return action_val

            result = self.tools[action_name](action_val)  #result must be a str, list, or tuple
            # Handle the case where 'passages' attribute is missing
            output[f"Observation_{hop+1}"] = getattr(result, "passages", result)

        except Exception:
            output[f"Observation_{hop+1}"] = (
                "Mislukt om de actie te parseren. Slechte opmaak of onjuiste actienaam."
            )
            # raise e

    def forward(self, **kwargs):
        args = {key: kwargs[key] for key in self.input_fields.keys() if key in kwargs}

        for hop in range(self.max_iters):
            # with dspy.settings.context(show_guidelines=(i <= 2)):
            output = self.react[hop](**args)
            output[f'Action_{hop + 1}'] = output[f'Action_{hop + 1}'].split('\n')[0]

            if action_val := self.act(output, hop):
                break
            args.update(output)

        observations = [args[key] for key in args if key.startswith("Observation")]
        
        # assumes only 1 output field for now - TODO: handling for multiple output fields
        return dspy.Prediction(observations=observations, **{list(self.output_fields.keys())[0]: action_val or ""})
