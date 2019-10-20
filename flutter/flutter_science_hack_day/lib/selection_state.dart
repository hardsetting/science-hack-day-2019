import 'package:flutter/widgets.dart';
import 'package:flutter_science_hack_day/model.dart';

class PokemonSelector with ChangeNotifier {
  static final Pokemon notSet =
      Pokemon("-1", "assets/question_mark.jpg", EnergyType.normal);

  Pokemon _pokemon;
  EnergyType _type;
  Pokemon _newPokemon = notSet;

  Pokemon getPokemon() => _pokemon;

  EnergyType getType() => _type;

  Pokemon getNewPokemon() {
    return _newPokemon;
  }

  void setPokemon(Pokemon pokemon) {
    if (this._pokemon == null) {
      this._pokemon = pokemon;
    } else if (this._pokemon.id == pokemon.id) {
      this._pokemon = null;
    } else {
      this._pokemon = pokemon;
    }
    updateNewPokemon();
    notifyListeners();
  }

  void setType(EnergyType type) {
    if (this._type == type) {
      this._type = null;
    } else {
      this._type = type;
    }
    updateNewPokemon();
    notifyListeners();
  }

  void updateNewPokemon() {
    if (this._pokemon == null || this._type == null) {
      _newPokemon = notSet;
    } else if (this._pokemon.type == this._type) {
      _newPokemon = this._pokemon;
    } else {
      final id = _pokemon.id;
      final type = this._type;
      // TODO(calderwoodra): call an API or use the model to generate the image
      Future.delayed(Duration(seconds: 2)).whenComplete(() {
        if (this._pokemon != null &&
            this._pokemon.id == id &&
            this._type == type) {
          _newPokemon = Pokemon(
              _pokemon.id + _type.toString(), "assets/squirtle.jpeg", _type);
          notifyListeners();
        }
      });
    }
  }
}
