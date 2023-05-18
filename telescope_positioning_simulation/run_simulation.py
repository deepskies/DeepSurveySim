import cleo

from telescope_positioning_simulation.Survey.survey import Survey
from telescope_positioning_simulation.IO.read_config import ReadConfig
from telescope_positioning_simulation.IO.save_simulation import SaveSimulation

# Ask for observatory config 
# Ask for survey config 

telescope_config = ReadConfig("", survey=False)
survey_config =  ReadConfig("", survey=True)

survey = Survey(survey_config, telescope_config)

survey_says = survey()

save = SaveSimulation(survey_instance=survey, survey_results=survey_says)

save()
