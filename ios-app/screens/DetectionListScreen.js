import React from 'react';
import { View, Text, StyleSheet, FlatList } from 'react-native';

const detections = [
  { id: '1', species: 'Fox', time: '2025-06-01 22:15' },
  { id: '2', species: 'Deer', time: '2025-06-01 22:30' },
  { id: '3', species: 'Owl', time: '2025-06-01 23:00' },
];

export default function DetectionListScreen() {
  const renderItem = ({ item }) => (
    <View style={styles.item}>
      <Text style={styles.species}>{item.species}</Text>
      <Text style={styles.time}>{item.time}</Text>
    </View>
  );

  return (
    <FlatList
      style={styles.list}
      data={detections}
      renderItem={renderItem}
      keyExtractor={(item) => item.id}
    />
  );
}

const styles = StyleSheet.create({
  list: {
    flex: 1,
    padding: 16,
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
