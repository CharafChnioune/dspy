import dspy
from dspy.predict.avatar.models import Action


class Acteur(dspy.Signature):
    """Je krijgt `Tools` die een lijst van tools zijn om het `Doel` te bereiken. Gezien de gebruikersquery is het jouw taak om te beslissen welke tool te gebruiken en welke invoerwaarden je aan de tool moet geven.

Je zult de actie outputten die nodig is om het `Doel` te bereiken. `Actie` moet een tool bevatten om te gebruiken en de invoerquery om aan de tool door te geven.

Opmerking: Je kunt ervoor kiezen om geen tools te gebruiken en direct het eindantwoord te geven. Je kunt ook één tool meerdere keren gebruiken met verschillende invoerqueries indien van toepassing."""

    doel: str = dspy.InputField(
        prefix="Doel:",
        desc="Taak die moet worden volbracht.",
    )
    tools: list[str] = dspy.InputField(
        prefix="Tools:",
        desc="lijst van te gebruiken tools",
    )
    actie_1: Action = dspy.OutputField(
        prefix="Actie 1:",
        desc="Eerste actie die moet worden ondernomen.",
    )
