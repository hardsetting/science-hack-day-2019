enum EnergyType {
  electric,
  fire,
  grass,
  psychic,
  normal,
  water
}


class Pokemon {

  final String id;
  final String url;
  final EnergyType type;

  Pokemon(this.id, this.url, this.type);
}

class Energy {
  final EnergyType type;
  final String url;

  Energy(this.type, this.url);
}

List<Pokemon> getPokemon() {
  // TODO(calderwoodra): call an actual API to get the real list of pokemon?
  return [
    Pokemon("001", "assets/bulbasaur.png", EnergyType.grass),
    Pokemon("004", "assets/charmander.png", EnergyType.fire),
    Pokemon("007", "assets/squirtle.jpeg", EnergyType.water),
    Pokemon("025", "assets/pikachu.png", EnergyType.electric),
  ];
}

List<Energy> getEnergy() {
  return [
    Energy(EnergyType.electric, "assets/electricity.png"),
    Energy(EnergyType.fire, "assets/fire.png"),
    Energy(EnergyType.grass, "assets/leaf.png"),
    Energy(EnergyType.psychic, "assets/psychic.png"),
    Energy(EnergyType.normal, "assets/normal.png"),
    Energy(EnergyType.water, "assets/water.png"),
  ];
}
