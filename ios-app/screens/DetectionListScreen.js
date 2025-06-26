import React, { useState } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, TextInput } from 'react-native';

const detections = [
  {
    id: '1',
    species: 'Fox',
    time: '2025-06-01 22:15',
    latitude: 37.78825,
    longitude: -122.4324,
    image: 'https://via.placeholder.com/400',
  },
  {
    id: '2',
    species: 'Deer',
    time: '2025-06-01 22:30',
    latitude: 37.78925,
    longitude: -122.4344,
    image: 'https://via.placeholder.com/400',
  },
  {
    id: '3',
    species: 'Owl',
    time: '2025-06-01 23:00',
    latitude: 37.79025,
    longitude: -122.4354,
    image: 'https://via.placeholder.com/400',
  },
];

export default function DetectionListScreen({ navigation }) {
  const [query, setQuery] = useState('');

  const filtered = detections.filter((d) =>
    d.species.toLowerCase().includes(query.toLowerCase())
  );

  const renderItem = ({ item }) => (
    <TouchableOpacity
      onPress={() => navigation.navigate('DetectionDetail', { detection: item })}
    >
      <View style={styles.item}>
        <Text style={styles.species}>{item.species}</Text>
        <Text style={styles.time}>{item.time}</Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.search}
        placeholder="Search species"
        value={query}
        onChangeText={setQuery}
      />
      <FlatList
        style={styles.list}
        data={filtered}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  search: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  list: {
    flex: 1,
    paddingHorizontal: 16,
  },
  item: {
    paddingVertical: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: '#ccc',
  },
  species: {
    fontSize: 16,
  },
  time: {
    color: '#555',
  },
});
