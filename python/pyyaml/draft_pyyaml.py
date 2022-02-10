'''https://pyyaml.org/wiki/PyYAMLDocumentation'''
import yaml

tmp1 = [
    'name: Vorlin Laruknuzum',
    'sex: Male',
    'class: Priest',
    'title: Acolyte',
    'hp: [32, 71]',
    'sp: [1, 13]',
    'gold: 423',
    'inventory:',
    '- a Holy Book of Prayers (Words of Wisdom)',
    '- an Azure Potion of Cure Light Wounds',
    '- a Silver Wand of Wonder',
]
x1 = yaml.load('\n'.join(tmp1))

x2 = {
    'name': "The Cloak 'Colluin'",
    'depth': 5,
    'rarity': 45,
    'weight': 10,
    'cost': 50000,
    'flags': ['INT', 'WIS', 'SPEED', 'STEALTH'],
}
yaml.dump(x2)
