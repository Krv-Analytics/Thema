# File: tests/multiverse/universe/test_galaxy.py
# Lasted Updated: 04-23-24
# Updated By: JW

import os

from thema.multiverse import Galaxy


class Test_Galaxy:
    """Pytest class for Galaxy"""

    def test_yaml_init(self, temp_galaxyYaml_1):
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)

    def test_fit(self, temp_galaxyYaml_1):
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()

    def test_collapse(self, temp_galaxyYaml_1):
        galaxy = Galaxy(YAML_PATH=temp_galaxyYaml_1.name)
        galaxy.fit()
        selection = galaxy.collapse()
        
        assert isinstance(selection, dict)
        for group, selected in selection.items():
            assert isinstance(selected, dict) 

            star = selected['star']
            cluster_size = selected['cluster_size']
            assert type(cluster_size) == int or type(cluster_size) == float
            assert os.path.exists(star)
        assert temp_galaxyYaml_1.name

        

	
