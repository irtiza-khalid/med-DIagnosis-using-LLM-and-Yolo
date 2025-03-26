# Medical Diagnosis Using LLM and YOLO

This Streamlit application utilizes Large Language Models (LLM) and YOLO (You Only Look Once) for medical image diagnosis. Users can upload medical images to receive diagnostic insights based on advanced machine learning models.

## Features

- **Image Upload:** Users can upload medical images for analysis.
- **Real-time Diagnosis:** The application provides instant diagnostic insights.
- **Interactive Visualizations:** The results include highlighted detected anomalies in images.

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/med-diagnosis-using-llm-and-yolo.git
cd med-diagnosis-using-llm-and-yolo
```

### Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Download Model Weights

Due to file size constraints, model weights are hosted externally. The application is configured to download these weights automatically from Google Drive. Ensure you have a stable internet connection during the initial run.

### Run the Application

Launch the Streamlit app with the following command:

```bash
streamlit run app.py
```

Access the application by navigating to [http://localhost:8501](http://localhost:8501) in your web browser.

## Usage

1. **Upload an Image:** Click on the "Browse files" button to select and upload a medical image.
2. **View Diagnosis:** The app will process the image and display diagnostic results, including any detected anomalies highlighted within the image.

## External Model Weights

The application utilizes pre-trained model weights that are not included in this repository due to their large size. These weights are hosted on Google Drive and are automatically downloaded when the application is first run. This approach ensures compliance with repository size limitations and streamlines the setup process.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository:** Click on the "Fork" button at the top right corner of this page.
2. **Create a New Branch:** Use a descriptive name for your branch.
3. **Make Changes:** Implement your features or bug fixes.
4. **Submit a Pull Request:** Provide a clear description of your changes and the problem they solve.

## Acknowledgements

- **Streamlit:** For providing an intuitive framework for building interactive web applications.
- **YOLO:** For the robust object detection model utilized in this application.
- **Community Contributors:** For their valuable inputs and feedback.

**Note:** This application is intended for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

