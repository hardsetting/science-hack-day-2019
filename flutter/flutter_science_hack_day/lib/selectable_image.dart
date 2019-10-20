import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_science_hack_day/model.dart';
import 'package:flutter_science_hack_day/selection_state.dart';
import 'package:provider/provider.dart';


class SelectableImage extends StatelessWidget {
  final size, url, expanded;
  final Pokemon pokemon;
  final EnergyType energyType;

  SelectableImage(this.url, this.size, {this.pokemon, this.energyType, this.expanded = false});

  @override
  Widget build(BuildContext context) {
    return expanded ? Expanded(child: _image(context)) : _image(context);
  }

  Widget _image(BuildContext context) {

    return GestureDetector(
      onTap: () {
        if (pokemon == null) {
          Provider.of<PokemonSelector>(context, listen: false).setType(energyType);
        } else {
          Provider.of<PokemonSelector>(context, listen: false).setPokemon(pokemon);
        }
      },
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(
            width: 2.0,
            color: Color.fromRGBO(255, 0, 0, isSelected(context) ? 1 : 0),
          ),
        ),
        child: Image.asset(
          url,
          height: size,
          width: size,
        ),
      ),
    );
  }

  bool isSelected(BuildContext context) {
    if (pokemon == null) {
      return energyType == Provider.of<PokemonSelector>(context).getType();
    } else {
      Pokemon providedPokemon = Provider.of<PokemonSelector>(context).getPokemon();
      return providedPokemon != null && pokemon.id == providedPokemon.id;
    }
  }
}
