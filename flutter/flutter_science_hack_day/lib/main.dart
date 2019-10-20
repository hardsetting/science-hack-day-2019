import 'package:flutter/material.dart';
import 'package:flutter_science_hack_day/model.dart';
import 'package:flutter_science_hack_day/selectable_image.dart';
import 'package:flutter_science_hack_day/selection_state.dart';
import 'package:provider/provider.dart';

void main() => runApp(MyApp());


class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Science Hack Day 2019',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(title: 'Science Hack Day 2019'),
    );
  }
}

class MyHomePage extends StatelessWidget {
  MyHomePage({Key key, this.title}) : super(key: key);

  final String title;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
      ),
      body: MultiProvider(
        providers: [
          ChangeNotifierProvider(builder: (_) => PokemonSelector()),
        ],
        child: Consumer<PokemonSelector>(
          builder: (context, counter, child) {
            return child;
          },
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                Expanded(child: pokemonList()),
                Row(children: firstEnergyRow()),
                Row(children: secondEnergyRow()),
                NewPokemonWidget(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget pokemonList() {
    return FutureBuilder<List<Pokemon>>(
      future: Future.delayed(Duration(seconds: 2), () => getPokemon()),
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          return SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(children: pokemon(snapshot.data)));
        } else {
          return Center(child: CircularProgressIndicator());
        }
      },
    );
  }

  List<Widget> pokemon(List<Pokemon> list) {
    return list
        .map((pokemon) => SelectableImage(pokemon.url, 150.0, pokemon: pokemon))
        .toList();
  }

  List<Widget> firstEnergyRow() {
    return getEnergy()
        .take(3)
        .map((energy) => SelectableImage(energy.url, 100.0,
            energyType: energy.type, expanded: true))
        .toList();
  }

  List<Widget> secondEnergyRow() {
    return getEnergy()
        .skip(3)
        .map((energy) => SelectableImage(energy.url, 100.0,
            energyType: energy.type, expanded: true))
        .toList();
  }
}

class NewPokemonWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Image.asset(
        Provider.of<PokemonSelector>(context).getNewPokemon().url,
        height: 150.0,
        width: 150.0,
      ),
    );
  }
}
