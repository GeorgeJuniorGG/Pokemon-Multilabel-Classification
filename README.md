# Pokémon Multilabel Classification

This is a Machine Learning project focused on **multilabel classification** of Pokémon. The goal is to predict types for each Pokémon using their images. The project includes data preprocessing, model training, evaluation, and documentation.

---

## 📁 **Project Structure**
- **`compiled_data/`**: Contains the image paths and their corresponding type relations.
- **`data/`**: Includes the Pokémon sprites and their labels, separated into `train`, `validation`, and `test` datasets.
- **`evaluation/`**: Script for evaluating Pokémon images against predicted types.
- **`pokedata.py`**: Script for data preprocessing, ensuring the data is ready for model training.
- **`Final report/`**: The final report for the Machine Learning course (in Portuguese).
- **`Images/`**: Visual resources related to the models and dataset.
- **`Models/`**: Contains model scripts, including:
  - `tds.py`
  - `tds Da.py`
  - `MobileNet.py`
- **`PokeAPI/`**: Data fetched from [PokeAPI](https://github.com/PokeAPI/pokeapi). Huge thanks to them for providing publicly available Pokémon data!
- **`Trained models/`**: Pre-trained model files in `.h5` format.

---

## 🛠️ **How to Use**
1. **Set Up Environment**:
   - Install the required Python libraries installed for Machine Learning and data processing.

2. **Prepare the Dataset**:
   - Use the scripts in `pokedata.py` to preprocess the data if needed.
   - Verify the data structure in the `data/` directory (organized into `train`, `validation`, and `test`).

3. **Train a Model**:
   - Run one of the model scripts (`tds.py`, `tds Da.py`, or `MobileNet.py`) to train your model on the provided dataset.

4. **Evaluate Results**:
   - Use the evaluation script in `evaluation/` to test the model and compare predictions against true labels.

---

## 🌟 **Acknowledgments**
- **PokeAPI**: Special thanks for making the Pokémon data publicly accessible. Visit [PokeAPI](https://pokeapi.co/) for more information.
- This project was completed as part of the Machine Learning course requirements.

---

## 📄 **Final Report**
The final report, written in Portuguese, can be found in the `Final report/` directory. It includes detailed explanations of the project objectives, methods, and outcomes.

---

## 📫 **Contact**
If you have questions or suggestions, feel free to reach out via [GitHub Issues](https://github.com/GeorgeJuniorGG/Pokemon-Multilabel-Classification/issues).
