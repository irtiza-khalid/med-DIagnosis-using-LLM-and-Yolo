import streamlit as st
import pydicom
import numpy as np
import cv2
import tempfile
import base64
import json
# Import your specialized modules
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from ultralytics import YOLO
import torch
# Ensure session state is initialized
if "responses" not in st.session_state:
    st.session_state.responses = {}

# Load the trained YOLOv8 model
model = YOLO('best.pt')
modelmri = YOLO('bestMRI.pt')



# Define the analysis query for the medical agent
analysis_query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links as well
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""
st.set_page_config(
    page_title="DICOM Analyzer",
    page_icon="🩻",  # Change to a local file path or a URL if needed

)


# Define Analysis Query Template
analysis_query_template = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and provide the requested section.

### {section_title}
{section_description}

Format your response using markdown headers and bullet points. Be concise yet thorough.
"""

# Define Analysis Sections
sections = {
    "Image Type & Region": "Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)\n- Identify the patient's anatomical region and positioning\n- Comment on image quality and technical adequacy.",
    "Key Findings": "List primary observations systematically.\n- Note any abnormalities with precise descriptions.\n- Include measurements and densities where relevant.\n- Describe location, size, shape, and characteristics.\n- Rate severity: Normal/Mild/Moderate/Severe.",
    "Diagnostic Assessment": "Provide primary diagnosis with confidence level.\n- List differential diagnoses in order of likelihood.\n- Support each diagnosis with observed evidence from the patient's imaging.\n- Note any critical or urgent findings.",
    "Patient-Friendly Explanation": "Explain the findings in simple, clear language.\n- Avoid medical jargon or provide clear definitions.\n- Use visual analogies if helpful.\n- Address common patient concerns.",
    "Research Context": "Use the DuckDuckGo search tool to find:\n- Recent medical literature about similar cases.\n- Standard treatment protocols.\n- A list of relevant medical links.\n- Any relevant technological advances."
}

# Function: Encode image for Gemini API
def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
def route(image_path):
    image_data = encode_image(image_path)
    genai.configure(api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ")

    model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": 0.2})
    routing_prompt="""Examine the provided medical image and precisely determine its category: X-ray of bones or MRI of the brain.

    If the image predominantly displays skeletal structures, classify it as "bone".

    If it primarily depicts brain soft tissue, classify it as "brain".

    Your response must be a single word: "brain" or "bone"—no additional text, explanations, or symbols."""
   
    response = model.generate_content([routing_prompt, {"mime_type": "image/jpg", "data": image_data}])
    print(response.candidates[0].content.parts[0].text)
    return response.candidates[0].content.parts[0].text

# Function: Analyze Image for Affected Regions
def analyze_image(image_path,analysis):
    image_data = encode_image(image_path)
    genai.configure(api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ")

    model = genai.GenerativeModel("gemini-2.5-pro-preview-03-25", generation_config={"temperature": 0.7})

    analysis_prompt = f"""
       Analyze the provided DICOM image and identify any abnormalities. For each detected region, provide a JSON object with:
      - "region_description": Label of the abnormality (e.g., "lesion," "tumor," "fracture").
      - "bounding_box": Coordinates with "x_min," "y_min," "x_max," and "y_max" corresponding to the image's pixel positions.
      - "confidence_score": Detection certainty between 0 and 1.

      Ensure all coordinates precisely match the DICOM image's resolution.
      analysis of image: {analysis}
        ### **Expected Output Format:**

        ```json
        {{
          "affected_regions": [
            {{
              "region_description": "tumor",
              "bounding_box": {{
                "x_min": 150,
                "y_min": 200,
                "x_max": 300,
                "y_max": 350
              }},
              "confidence_score": 0.95
            }}
          ]
        }}
        ```

      """

    response = model.generate_content([analysis_prompt, {"mime_type": "image/jpg", "data": image_data}])

    response_text = response.candidates[0].content.parts[0].text
    start_index = response_text.find("```json")
    end_index = response_text.find("```", start_index + 7)

    if start_index != -1 and end_index != -1:
        json_str = response_text[start_index + 7 : end_index].strip()
        return json.loads(json_str)
    return None

# Function: Draw Bounding Box on Image
def draw_bounding_boxes(image_path, analysis_json, output_path):
    image = cv2.imread(image_path)

    for region in analysis_json.get("affected_regions", []):
        bbox = region["bounding_box"]
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Draw circle at center
        x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2
        cv2.circle(image, (x_center, y_center), radius=64, color=(255, 0, 0), thickness=2)

    cv2.imwrite(output_path, image)
    return output_path
def main():
    st.title("🩺 Medical Image Analysis")
    st.write("Yo! Upload your DICOM file via the sidebar and select the slice you want to analyze using the buttons below the image.")

    # Upload functionality in the sidebar

    uploaded_file = st.sidebar.file_uploader("Upload a medical image (DICOM, PNG, JPEG)",type=["dcm", "DCM", "dicom", "tif", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.responses.clear()  # Clear previous responses
            st.session_state.last_uploaded_file = uploaded_file.name  # Store current file name

    if uploaded_file is not None:
        try:
            # Get the file name and extension
            file_name = uploaded_file.name
            file_extension = file_name.split(".")[-1].lower()
            if file_extension in ["dcm", "dicom", "tif"]:
                # Read the DICOM file directly from the uploaded file
              dicom_data = pydicom.dcmread(uploaded_file)
              st.success("DICOM file uploaded successfully!")


              if 'PixelData' in dicom_data:
                  image_data = dicom_data.pixel_array
                  total=image_data.shape[0]
                  # For 3D images, initialize session state for slice index
                  if len(image_data.shape) == 3:
                      if "slice_idx" not in st.session_state:
                          st.session_state.slice_idx = 0

                      # Display the currently selected slice
                      selected_slice = image_data[st.session_state.slice_idx]

                      # Normalize the image for display if necessary
                      if selected_slice.dtype != "uint8":
                          selected_slice = cv2.normalize(selected_slice, None, 0, 255, cv2.NORM_MINMAX)
                          selected_slice = np.uint8(selected_slice)

                      st.image(selected_slice, caption=f"Slice {st.session_state.slice_idx + 1}  of Total {total}", use_container_width=True)

                      # Create Previous and Next buttons below the image
                      # Create a row with three equally sized columns
                      # Create three columns with equal width
                      col_prev, col_analyze, col_next = st.columns([2.9, 3, 1])

                      # Define the buttons within their respective columns
                      with col_prev:
                          prev_clicked = st.button("Previous")
                      with col_analyze:
                          analyze_clicked = st.button("Analyze")
                      with col_next:
                          next_clicked = st.button("Next")
                          # Add your new button under the Analyze button
                      with col_analyze:
                          st.write("")  # Adds a bit of space
                          new_button_clicked = st.button("Analyze ALL")
                      # Handle the button clicks outside the column context
                      if prev_clicked:
                          if st.session_state.slice_idx > 0:
                              st.session_state.slice_idx -= 1
                          st.rerun()

                      if next_clicked:
                          if st.session_state.slice_idx < image_data.shape[0] - 1:
                              st.session_state.slice_idx += 1
                          st.rerun()

                      if analyze_clicked:
                              with st.spinner("Analyzing..."):
                                  # Save the selected slice to a temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                                  temp_image_path = temp_file.name

                                  min_val = np.min(selected_slice)
                                  max_val = np.max(selected_slice)

                                  normalized_image = ((selected_slice - min_val) / (max_val - min_val) * 255).astype(np.uint8)


                                  cv2.imwrite(temp_image_path, normalized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


                                  # Initialize the medical agent (replace the API key with your own)
                                  medical_agent = Agent(
                                      model=Gemini(
                                          api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # <-- Replace with your Gemini API key
                                          id="gemini-2.0-flash"
                                      ),
                                      tools=[DuckDuckGo()],
                                      markdown=True
                                  )

                                  # Run analysis on the selected slice image
                                  response = medical_agent.run(analysis_query, images=[temp_image_path])
                                  st.markdown("### Analysis for Selected Slice")
                                  st.write(response.content)

                                  # Optionally, generate an overall summary using the LLM
                                  llm = ChatGoogleGenerativeAI(
                                      model="gemini-2.0-flash",
                                      api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # <-- Replace with your Gemini API key
                                      temperature=0.0
                                  )
                                  summary_prompt = f"""
                                  Here is the analysis of the selected slice:
                                  {response.content}

                                  Please provide a complete summary and overall diagnosis based on the analysis.
                                  """
                                  summary_response = llm.invoke(summary_prompt)
                                  st.markdown("### Overall Summary & Diagnosis")
                                  st.write(summary_response.content)
                      elif new_button_clicked:
                        # Logic when "New Button" is clicked
                        with st.spinner("Analyzing ALL DCM images..."):
                          if 'PixelData' in dicom_data:
                              image_data = dicom_data.pixel_array  # Convert pixel data to numpy array
                              #st.write(f"Total Slices: {len(image_data)}")  # Show slice count

                              if len(image_data.shape) == 3:  # Check for 3D image (CT/MRI)
                                  slice_image_paths = []
                                  with st.spinner("Processing all slices..."):
                                      for i in range(len(image_data)):
                                          slice_data = image_data[i]
                                          #st.write(f"Processing Slice {i + 1} with shape: {slice_data.shape}")

                                          # Normalize the image data
                                          min_val = np.min(slice_data)
                                          max_val = np.max(slice_data)
                                          normalized_image = ((slice_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

                                          # Save slice as temp image
                                          with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                                              temp_image_path = temp_file.name
                                              cv2.imwrite(temp_image_path, normalized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                                              slice_image_paths.append(temp_image_path)

                                          # Display image in Streamlit
                                          #st.image(temp_image_path, caption=f"Slice {i + 1}", use_column_width=True)

                                  # Send images to medical agent for analysis
                                  if slice_image_paths:
                                    # Initialize medical agent
                                      medical_agent = Agent(
                                        model=Gemini(
                                            api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                                            id="gemini-2.0-flash"
                                        ),
                                        tools=[DuckDuckGo()],
                                        markdown=True
                                      )
                                      st.write(f"Total Slices: {len(slice_image_paths)}")
                                      response = medical_agent.run(analysis_query, images=slice_image_paths)
                                      st.markdown("### Analysis Results for All Slices")
                                      st.write(response.content)
                                      # Optional: Generate overall summary and diagnosis
                                      llm = ChatGoogleGenerativeAI(
                                          model="gemini-2.0-flash",
                                          api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                                          temperature=0.0
                                      )
                                      summary_prompt = f"""
                                      Here is the analysis of the medical image:
                                      {response.content}

                                      Please provide a complete summary and overall diagnosis based on the analysis.
                                      """

                                      summary_response = llm.invoke(summary_prompt)
                                      st.markdown("### Overall Summary & Diagnosis")
                                      st.write(summary_response.content)
                          else:
                              st.write("No valid DICOM data found.")

                  else:
                      st.write("The uploaded DICOM file contains 2D image data.")
                      normalized_image = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
                      # Apply histogram equalization for better contrast
                      #equalized_image = cv2.equalizeHist(image_data.astype(np.uint8))
                      #st.image(normalized_image, caption="DICOM Image", use_column_width=True)
                      col1, col2 = st.columns(2)
                      with col1:
                          st.image(normalized_image, caption="📷 Original Image", use_container_width=True)

                          # Save Image Temporarily for AI Processing
                      with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                          temp_image_path = temp_file.name
                          cv2.imwrite(temp_image_path, normalized_image.astype("uint8"), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                        # Initialize AI Medical Agent
                      medical_agent = Agent(
                          model=Gemini(api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ", id="gemini-2.0-flash"),
                          tools=[DuckDuckGo()],
                          markdown=True
                      )
                      image_query = analysis_query_template.format(
                                  section_title="Image Type & Region",
                                  section_description=sections["Image Type & Region"]
                              )
                      response = medical_agent.run(image_query, images=[temp_image_path])
                      analysis_json = analyze_image(temp_image_path,response)
                      marked_image_path = "marked_image.jpg"
                      marked_path = draw_bounding_boxes(temp_image_path, analysis_json, marked_image_path)

                      # **Show the Marked Image in the Second Column**
                      with col2:
                          st.image(marked_path, caption="🔴 Marked Affected Regions", use_container_width=True)



                        # Run "Image Type & Region" analysis only once
                      if "Image Type & Region" not in st.session_state.responses:
                          with st.spinner("Analyzing Image Type & Region..."):
                              image_query = analysis_query_template.format(
                                  section_title="Image Type & Region",
                                  section_description=sections["Image Type & Region"]
                              )
                              response = medical_agent.run(image_query, images=[temp_image_path])
                              st.session_state.responses["Image Type & Region"] = response.content if hasattr(response, 'content') else "No response received."

                      # Display "Image Type & Region" without rerunning it
                      st.markdown("### 1. Image Type & Region")
                      st.write(st.session_state.responses["Image Type & Region"])

                        # Buttons for Other Sections
                      for section_title, section_description in sections.items():
                          if section_title != "Image Type & Region":  # Exclude already displayed section
                              if st.button(f"Show {section_title}"):
                                  with st.spinner(f"Fetching {section_title}..."):
                                      section_query = analysis_query_template.format(
                                          section_title=section_title,
                                          section_description=section_description
                                      )
                                      response = medical_agent.run(section_query, images=[temp_image_path])
                                      st.markdown(f"### {section_title}")
                                      st.write(response.content if hasattr(response, 'content') else "No response received.")

                      # Run analysis with image
                      response = medical_agent.run(analysis_query, images=[temp_image_path])
                      #st.markdown("### Analysis Results")
                      #st.write(response.content)

                      # Optional: Generate overall summary and diagnosis
                      llm = ChatGoogleGenerativeAI(
                          model="gemini-2.0-flash",
                          api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                          temperature=0.0
                      )
                      summary_prompt = f"""
                      Here is the analysis of the medical image:
                      {response.content}

                      Please provide a complete summary and overall diagnosis based on the analysis.
                      """

                      summary_response = llm.invoke(summary_prompt)
                      st.markdown("### Overall Summary & Diagnosis")
                      st.write(summary_response.content)



              else:
                  st.error("No pixel data found in the DICOM file.")

            elif file_extension in ["png", "jpg", "jpeg"]:
              #file_bytes = uploaded_file.read()
              with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                  temp_file.write(uploaded_file.read())
                  temp_image_path = temp_file.name  # Store temp file path
              
                
              # Read the image using cv2.imread()
              image = cv2.imread(temp_image_path)
              image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              st.success("Image uploaded successfully!")

              col1, col2 = st.columns(2)
              with col1:
                  st.image(image_rgb, caption="📷 Uploaded Image", use_container_width=True)

                  # Save Image Temporarily for AI Processing
              suffix = f".{file_extension}"
              with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_image_path = temp_file.name
                if file_extension == "png":
                    cv2.imwrite(temp_image_path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                else:
                    cv2.imwrite(temp_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
              

                # Initialize AI Medical Agent
              medical_agent = Agent(
                  model=Gemini(api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ", id="gemini-2.0-flash"),
                  tools=[DuckDuckGo()],
                  markdown=True
              )
              #image_query = analysis_query_template.format(
              #            section_title="Image Type & Region",
              #            section_description=sections["Image Type & Region"]
              #        )
              #response = medical_agent.run(image_query, images=[temp_image_path])
              #analysis_json = analyze_image(temp_image_path,response)
              #marked_image_path = "marked_image.jpg"
              #marked_path = draw_bounding_boxes(temp_image_path, analysis_json, marked_image_path)

              # **Show the Marked Image in the Second Column**
              #with col2:
                  #st.image(marked_path, caption="🔴 Marked Affected Regions", use_container_width=True)
              route1=route(temp_image_path).strip()
              st.write(route1)
              if route1=="bone":
                # Perform segmentation using YOLOv8
                results = model.predict(source=temp_image_path, task='segment', mode='predict')

                # Save and display segmented image
                for result in results:
                    segmented_image = result.plot()  # Convert result to an image with bounding boxes/masks
                    segmented_path = "segmented_output.jpg"
                    cv2.imwrite(segmented_path, segmented_image)

                    with col2:
                        st.image(segmented_path, caption="🔴BONE Marked Affected Regions", use_container_width=True)
              else:
                # Perform segmentation using YOLOv8
                results = modelmri.predict(source=temp_image_path, task='segment', mode='predict')

                # Save and display segmented image
                for result in results:
                    segmented_image = result.plot()  # Convert result to an image with bounding boxes/masks
                    segmented_path = "segmented_output.jpg"
                    cv2.imwrite(segmented_path, segmented_image)

                    with col2:
                        st.image(segmented_path, caption="🔴 MRI Marked Affected Regions", use_container_width=True)




                # Run "Image Type & Region" analysis only once
              if "Image Type & Region" not in st.session_state.responses:
                  with st.spinner("Analyzing Image Type & Region..."):
                      image_query = analysis_query_template.format(
                          section_title="Image Type & Region",
                          section_description=sections["Image Type & Region"]
                      )
                      response = medical_agent.run(image_query, images=[temp_image_path])
                      st.session_state.responses["Image Type & Region"] = response.content if hasattr(response, 'content') else "No response received."

              # Display "Image Type & Region" without rerunning it
              st.markdown("### 1. Image Type & Region")
              st.write(st.session_state.responses["Image Type & Region"])

              # Buttons for Other Sections
              for section_title, section_description in sections.items():
                  if section_title != "Image Type & Region":  # Exclude already displayed section
                      if st.button(f"Show {section_title}"):
                          with st.spinner(f"Fetching {section_title}..."):
                              section_query = analysis_query_template.format(
                                  section_title=section_title,
                                  section_description=section_description
                              )
                              response = medical_agent.run(section_query, images=[temp_image_path])
                              st.markdown(f"### {section_title}")
                              st.write(response.content if hasattr(response, 'content') else "No response received.")

              # Run analysis with image
              response = medical_agent.run(analysis_query, images=[temp_image_path])
              #st.markdown("### Analysis Results")
              #st.write(response.content)

              # Optional: Generate overall summary and diagnosis
              llm = ChatGoogleGenerativeAI(
                  model="gemini-2.0-flash",
                  api_key="AIzaSyBe5hCcwzCBrR1yeMMxh5ElHhvYPaqbLTQ",  # Replace with your own Gemini API key
                  temperature=0.0
              )
              summary_prompt = f"""
              Here is the analysis of the medical image:
              {response.content}

              Please provide a complete summary and overall diagnosis based on the analysis.
              """

              summary_response = llm.invoke(summary_prompt)
              st.markdown("### Overall Summary & Diagnosis")
              st.write(summary_response.content)



        except Exception as e:
            st.error(f"Error processing DICOM file: {e}")
        # Add company name at the bottom right corner
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 18px;
                right: 14px;
                font-size: 15px;
                color: gray;
            }
        </style>
        <div class="footer">
            Developed by <strong>PACE TECHNOLOGIES</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()
