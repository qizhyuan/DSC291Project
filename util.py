import numpy as np
from tqdm import tqdm


def downsampling(df, rating_matrix):
    users_interaction = []
    print("Downsampling")
    for index, row in tqdm(df.iterrows()):
        user, item = row["user"], row["item"]
        user_inter = [item]
        random_indices = [i for i in range(rating_matrix.shape[0]) if i != item]
        while len(user_inter) < 100:
            random_index = np.random.choice(random_indices, size=1)[0]
            random_indices.remove(random_index)
            if rating_matrix[random_index][user] == 0:
                user_inter.append(random_index)
        users_interaction.append((user, user_inter))

    return users_interaction
