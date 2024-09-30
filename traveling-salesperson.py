import numpy as np
import random

# Load city coordinates from the file (You can download the file manually)
def load_cities(filename):
    cities = np.loadtxt(filename)
    return cities

# Compute Euclidean distance between two cities
def euclidean_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# Total tour distance for a given permutation of cities
def total_distance(tour, cities):
    dist = 0
    for i in range(len(tour) - 1):
        dist += euclidean_distance(cities[tour[i]], cities[tour[i + 1]])
    # Return to the starting city
    dist += euclidean_distance(cities[tour[-1]], cities[tour[0]])
    return dist

# Initialize population with random tours
def initialize_population(pop_size, num_cities):
    population = [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]
    return population

# Selection (Tournament Selection)
def tournament_selection(population, cities, tournament_size=3):
    best = None
    for _ in range(tournament_size):
        individual = random.choice(population)
        if best is None or total_distance(individual, cities) < total_distance(best, cities):
            best = individual
    return best

# Crossover (Order Crossover OX)
def order_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]

    pointer = end
    for city in parent2:
        if city not in child:
            if pointer >= size:
                pointer = 0
            child[pointer] = city
            pointer += 1

    return child

# Mutation (Swap Mutation)
def swap_mutation(tour):
    idx1, idx2 = random.sample(range(len(tour)), 2)
    tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
    return tour

# Main evolutionary algorithm
def evolutionary_algorithm(cities, pop_size=100, generations=500, mutation_rate=0.1):
    num_cities = len(cities)
    population = initialize_population(pop_size, num_cities)

    for generation in range(generations):
        new_population = []

        for _ in range(pop_size):
            # Selection
            parent1 = tournament_selection(population, cities)
            parent2 = tournament_selection(population, cities)

            # Crossover
            child = order_crossover(parent1, parent2)

            # Mutation
            if random.random() < mutation_rate:
                child = swap_mutation(child)

            new_population.append(child)

        # Replace the old population with the new one
        population = new_population

        # Keep track of the best solution
        best_solution = min(population, key=lambda tour: total_distance(tour, cities))
        best_distance = total_distance(best_solution, cities)
        print(f"Generation {generation}: Best distance = {best_distance}")

    return best_solution, best_distance

if __name__ == "__main__":
    # Load the city data
    cities = load_cities("TSPDATA.txt")
    
    # Run the evolutionary algorithm
    best_tour, best_distance = evolutionary_algorithm(cities)
    
    # Output the best result
    print(f"Best tour: {best_tour}")
    print(f"Best distance: {best_distance}")