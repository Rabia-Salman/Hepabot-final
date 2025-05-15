import streamlit as st
import json
import pandas as pd
import tempfile
from enhanced_extraction import process_pdf
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
import base64
from io import BytesIO
from typing import List
from audio_transcription import process_audio_to_pdf
from dotenv import load_dotenv
import os
from vector_db import initialize_vector_db, add_patient_record
from extract_each_patient_json import save_patient_records
from utils import process_documents, create_vector_db, load_vector_db, create_rag_chain, ask_question


load_dotenv()

try:
    from elevenlabs import Voice, VoiceSettings, generate, play
    from elevenlabs.api import User
    ELEVENLABS_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {e}")
    ELEVENLABS_AVAILABLE = False

DB_PATH = "db/vector_db"
COLLECTION_NAME = "docs-hepabot-rag"
MODEL_NAME = "gpt-4o-mini"
ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.json']
input_json_path = "doctor_patient_data_80.json"
output_directory = "split_patient_files"

st.set_page_config(
    page_title="HEPABOT",
    page_icon="🏥",
    layout="wide"
)


@st.cache_data
def load_metadata(_refresh=False):
    try:
        with open(input_json_path, "r") as f:
            data = json.load(f)

        metadata_list = []
        for entry in data:
            if 'structured_data' in entry and 'patient_id' in entry:
                meta = entry['structured_data'].get('PatientDemographics', {})
                meta['patient_id'] = entry['patient_id']
                metadata_list.append(meta)
            else:
                print(f"Skipping entry missing required fields: {entry}")

        return metadata_list
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return []


@st.cache_resource
def get_vector_db():
    collection = initialize_vector_db()
    return collection


def format_search_results(results):
    formatted_results = []

    if results and 'documents' in results and results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if 'metadatas' in results else {}

            formatted_results.append({
                "patient_id": metadata.get("patient_id", "Unknown"),
                "gender": metadata.get("gender", "Unknown"),
                "age": metadata.get("age", "Unknown"),
                "mrn": metadata.get("mrn", "Unknown"),
                "diagnosis": metadata.get("diagnosis", "Unknown"),
                "content": doc
            })

    return formatted_results


def main():
    st.title("🏥 HEPABOT")

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "search"
    if 'refresh_data' not in st.session_state:
        st.session_state.refresh_data = False

    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        if st.button("Advance Search", use_container_width=True):
            st.session_state.page = "search"
        if st.button("Patient Browser", use_container_width=True):
            st.session_state.page = "browser"
        if st.button("Analytics", use_container_width=True):
            st.session_state.page = "analytics"
        if st.button("Generate Report", use_container_width=True):
            st.session_state.page = "generate_report"
        if st.button("Clinical Assistant", use_container_width=True):
            st.session_state.page = "clinical_assistant"

    # Page selection
    if st.session_state.page == "search":
        show_search_page()
    elif st.session_state.page == "browser":
        show_browser_page()
    elif st.session_state.page == "analytics":
        show_analytics_page()
    elif st.session_state.page == "generate_report":
        show_generate_report_page()
    elif st.session_state.page == "clinical_assistant":
        show_clinical_assistant_page()



def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded files and return their paths"""
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension not in ALLOWED_EXTENSIONS:
            st.warning(f"Unsupported file format: {file_extension}. Skipping {uploaded_file.name}")
            continue

        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(temp_file_path)

    return file_paths

def text_to_speech(text: str, api_key: str = None) -> BytesIO:
    """Convert text to speech using ElevenLabs API"""
    if not ELEVENLABS_AVAILABLE:
        st.error("ElevenLabs package is not installed. Voice over is not available.")
        return None

    if not api_key:
        api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        st.error("ElevenLabs API key is not set. Voice over is not available.")
        return None

    try:
        import elevenlabs
        elevenlabs.set_api_key(api_key)

        voices = elevenlabs.voices()
        if not voices:
            st.error("No voices available in your ElevenLabs account")
            return None
        voice_id = voices[0].voice_id

        audio = elevenlabs.generate(
            text=text,
            voice=st.session_state.get("selected_voice_id", voice_id),
            model="eleven_turbo_v2"
        )

        return BytesIO(audio)
    except Exception as e:
        st.error(f"Error generating voice: {str(e)}")
        return None

def get_download_link(data, filename, text):
    """Generate a download link for a file"""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href


def show_clinical_assistant_page():
    st.header("🩺 Clinical Assistant")
    st.write("Upload medical documents and ask questions to get AI-assisted medical insights.")

    # Initialize session state variables
    if 'vector_db_created' not in st.session_state:
        st.session_state.vector_db_created = False
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'last_response' not in st.session_state:
        st.session_state.last_response = ""
    if 'selected_voice_id' not in st.session_state:
        st.session_state.selected_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice ID (Adam)

    if os.path.exists(DB_PATH) and not st.session_state.vector_db_created:
        st.session_state.vector_db_created = True

    with st.sidebar:
        st.header("Document Management")

        db_exists = os.path.exists(DB_PATH)
        if db_exists:
            st.success("Vector database exists! Ready to answer questions.")
        else:
            st.warning("No vector database found. Please upload documents.")

        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or JSON files",
            accept_multiple_files=True,
            type=["pdf", "txt", "json"]
        )

        # Database creation options
        with st.expander("Advanced Options", expanded=False):
            chunk_size = st.number_input("Chunk Size", value=1200, min_value=500, max_value=2000)
            chunk_overlap = st.number_input("Chunk Overlap", value=300, min_value=0, max_value=500)

        if st.button("Process Documents & Create Database"):
            if not uploaded_files:
                st.error("Please upload at least one document.")
            else:
                with st.status("Processing documents..."):
                    file_paths = save_uploaded_files(uploaded_files)

                    if file_paths:
                        st.text(f"Processing {len(file_paths)} documents...")
                        docs = process_documents(
                            file_paths,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )

                        st.text("Creating vector database...")
                        vector_db = create_vector_db(
                            docs,
                            persist_directory=DB_PATH,
                            collection_name=COLLECTION_NAME
                        )

                        st.session_state.vector_db_created = True
                        st.success(f"Vector database created with {len(docs)} chunks!")
                    else:
                        st.error("No valid documents were uploaded.")

        # Delete database button
        if st.button("Delete Database"):
            if os.path.exists(DB_PATH):
                import shutil
                shutil.rmtree(DB_PATH)
                st.session_state.vector_db_created = False
                st.session_state.rag_chain = None
                st.success("Database deleted successfully.")
            else:
                st.info("No database to delete.")

    st.subheader("Medical Diagnosis Assistant")
    if st.session_state.vector_db_created and not st.session_state.rag_chain:
        with st.status("Loading RAG chain..."):
            try:
                vector_db = load_vector_db(
                    persist_directory=DB_PATH,
                    collection_name=COLLECTION_NAME
                )

                if vector_db:
                    st.session_state.rag_chain = create_rag_chain(vector_db, MODEL_NAME)
                    st.success("Ready to answer your medical questions!")
                else:
                    st.error("Failed to load vector database. Please create a new one.")
            except Exception as e:
                st.error(f"Error loading database: {str(e)}")
                st.session_state.vector_db_created = False

    if st.session_state.vector_db_created:
        col1, col2 = st.columns([3, 1])

        with col1:
            question = st.text_area(
                "Enter your medical question:",
                height=100,
                placeholder="Example: Patient has age 70 and shows symptoms of nausea and abdominal pain with fatigue, find disease"
            )

        with col2:
            voice_enabled = st.checkbox("Enable voice output", value=ELEVENLABS_AVAILABLE)

            if voice_enabled:
                if not ELEVENLABS_AVAILABLE:
                    st.warning("ElevenLabs package is not installed. Voice output disabled.")
                    voice_enabled = False
                else:
                    api_key = None
                    if not os.getenv("ELEVENLABS_API_KEY"):
                        api_key = st.text_input("ElevenLabs API Key", type="password")
                        if api_key:
                            os.environ["ELEVENLABS_API_KEY"] = api_key

                    # Show available voices if API key is provided
                    if api_key or os.getenv("ELEVENLABS_API_KEY"):
                        try:
                            import elevenlabs
                            elevenlabs.set_api_key(api_key or os.getenv("ELEVENLABS_API_KEY"))
                            voices = elevenlabs.voices()
                            if voices:
                                voice_options = {voice.name: voice.voice_id for voice in voices}
                                selected_voice = st.selectbox("Select voice", options=list(voice_options.keys()))
                                st.session_state.selected_voice_id = voice_options[selected_voice]
                            else:
                                st.info("No custom voices found. Will use default voice.")
                        except Exception:
                            st.info("Could not fetch voices. Will use default voice.")

        # Submit button
        if st.button("Get Diagnosis"):
            if not question:
                st.warning("Please enter a question.")
            elif not st.session_state.rag_chain:
                st.error("RAG chain is not loaded. Please create or load a database first.")
            else:
                with st.status("Generating answer..."):
                    try:
                        # Get answer from RAG chain
                        response = ask_question(st.session_state.rag_chain, question)
                        st.session_state.last_response = response
                    except Exception as e:
                        st.error(f"Error generating diagnosis: {str(e)}")

        # Display the response
        if st.session_state.last_response:
            st.subheader("Diagnosis Result:")
            st.markdown(st.session_state.last_response)

            col1, col2 = st.columns(2)

            # Download button
            with col1:
                if st.button("Download Result"):
                    download_data = st.session_state.last_response.encode()
                    st.markdown(
                        get_download_link(
                            download_data,
                            "diagnosis_result.txt",
                            "Download Diagnosis Result"
                        ),
                        unsafe_allow_html=True
                    )

            # Voice playback
            with col2:
                if voice_enabled and st.button("Play Voice"):
                    with st.spinner("Generating voice..."):
                        audio_data = text_to_speech(st.session_state.last_response, api_key)
                        if audio_data:
                            st.audio(audio_data, format='audio/mp3')
    else:
        st.info("Please upload documents and create a vector database to start asking questions.")


def show_search_page():
    st.header("Diagnosis Search")
    st.write("Search for patients with specific diagnoses or conditions")

    # Clear cache if refresh is needed
    if st.session_state.refresh_data:
        st.cache_data.clear()
        st.session_state.refresh_data = False

    # Load metadata
    metadata_list = load_metadata(_refresh=st.session_state.refresh_data)

    # Get unique diagnoses from the metadata
    all_diagnoses = []
    for meta in metadata_list:
        diagnosis = meta.get('Diagnosis', '')
        if diagnosis and diagnosis not in all_diagnoses:
            all_diagnoses.append(diagnosis)

    # Sort diagnoses alphabetically
    all_diagnoses.sort()

    # Add an "All" option
    all_diagnoses = ["All"] + all_diagnoses

    # Create search interface
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input("Search by keyword in diagnosis:",
                                     placeholder="E.g., cancer, diabetes, heart")

    with col2:
        selected_diagnosis = st.selectbox("Filter by diagnosis:", all_diagnoses)

    # Filter options
    with st.expander("Advanced Filters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender_filter = st.selectbox("Gender", ["Any", "Male", "Female"])

        with col2:
            age_range = st.slider("Age Range", 0, 100, (0, 100))

        with col3:
            sort_by = st.selectbox("Sort by", ["Diagnosis", "Age", "MRN"])

    filtered_patients = []

    for patient in metadata_list:
        diagnosis = patient.get('Diagnosis', '')
        include = True

        if selected_diagnosis != "All" and diagnosis != selected_diagnosis:
            include = False

        if search_query and search_query.lower() not in diagnosis.lower():
            include = False

        if gender_filter != "Any" and patient.get('Gender', '') != gender_filter:
            include = False

        try:
            patient_age = int(patient.get('Age', 0))
            if patient_age < age_range[0] or patient_age > age_range[1]:
                include = False
        except (ValueError, TypeError):
            pass

        if include:
            filtered_patients.append(patient)

    if sort_by == "Diagnosis":
        filtered_patients.sort(key=lambda x: x.get('Diagnosis', ''))
    elif sort_by == "Age":
        filtered_patients.sort(key=lambda x: int(x.get('Age', 0)) if x.get('Age', '').isdigit() else 0)
    elif sort_by == "MRN":
        filtered_patients.sort(key=lambda x: x.get('MRN', ''))


    if filtered_patients:
        df_display = []
        for patient in filtered_patients:
            df_display.append({
                "Patient ID": patient.get('patient_id', 'Unknown'),
                "MRN": patient.get('MRN', 'Unknown'),
                "Gender": patient.get('Gender', 'Unknown'),
                "Age": patient.get('Age', 'Unknown'),
                "Diagnosis": patient.get('Diagnosis', 'Unknown')
            })

        # Display as a table with ability to select a row
        df = pd.DataFrame(df_display)
        selected_indices = st.dataframe(df, use_container_width=True, height=300)

        # Allow selecting a patient for detailed view
        selected_patient_id = st.selectbox("Select a patient to view details:",
                                           [''] + [p.get('patient_id', '') for p in filtered_patients])

        if selected_patient_id:
            try:
                with open(input_json_path, "r") as f:
                    all_data = json.load(f)

                # Find the selected patient record
                patient_record = next((item for item in all_data if item.get('patient_id') == selected_patient_id),
                                      None)

                # Display patient Details
                if patient_record:
                    st.subheader("Patient Details")

                    # Get demographics
                    demographics = patient_record.get('structured_data', {}).get('PatientDemographics', {})

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MRN", demographics.get('MRN', 'N/A'))
                    with col2:
                        st.metric("Age", demographics.get('Age', 'N/A'))
                    with col3:
                        st.metric("Gender", demographics.get('Gender', 'N/A'))
                    with col4:
                        st.metric("Diagnosis", demographics.get('Diagnosis', 'N/A'))

                    # Display tabs for different sections of structured data
                    tabs = st.tabs(["Summary Report", "Clinical Summary", "Diagnostic Information", "Treatment Plan",
                                    "Conversation"])
                    with tabs[0]:
                        summary_report = patient_record.get('structured_data', {})
                        st.subheader("Summary Report")
                        st.write(summary_report)

                    with tabs[1]:
                        clinical_summary = patient_record.get('structured_data', {}).get('ClinicalSummary', {})

                        st.subheader("Clinical Summary")

                        # Active symptoms
                        st.write("**Active Symptoms:**")
                        symptoms = clinical_summary.get('ActiveSymptoms', [])
                        if symptoms:
                            for symptom in symptoms:
                                st.write(f"- {symptom}")
                        else:
                            st.write("No active symptoms recorded")

                        # Negative findings
                        st.write("**Negative Findings:**")
                        neg_findings = clinical_summary.get('NegativeFindings', [])
                        if neg_findings:
                            for finding in neg_findings:
                                st.write(f"- {finding}")
                        else:
                            st.write("No negative findings recorded")

                    with tabs[2]:
                        # Display diagnostic information
                        st.subheader("Diagnostic Information")

                        # Diagnostic conclusions
                        diag_conclusions = patient_record.get('structured_data', {}).get('DiagnosticConclusions', [])
                        if diag_conclusions:
                            st.write("**Diagnostic Conclusions:**")
                            for conclusion in diag_conclusions:
                                st.write(f"- {conclusion}")

                        # Diagnostic evidence
                        diag_evidence = patient_record.get('structured_data', {}).get('DiagnosticEvidence', {})

                        # Imaging findings
                        img_findings = diag_evidence.get('ImagingFindings', [])
                        if img_findings:
                            st.write("**Imaging Findings:**")
                            for finding in img_findings:
                                st.write(f"- {finding}")

                        # Lab results
                        lab_results = diag_evidence.get('LabResults', [])
                        if lab_results:
                            st.write("**Laboratory Results:**")
                            for result in lab_results:
                                st.write(f"- {result}")

                        # Pathology findings
                        path_findings = diag_evidence.get('PathologyFindings', [])
                        if path_findings:
                            st.write("**Pathology Findings:**")
                            for finding in path_findings:
                                st.write(f"- {finding}")

                        # Chronic conditions
                        chronic = patient_record.get('structured_data', {}).get('ChronicConditions', {})

                        # Chronic diseases
                        diseases = chronic.get('ChronicDiseases', [])
                        if diseases:
                            st.write("**Chronic Diseases:**")
                            for disease in diseases:
                                st.write(f"- {disease}")

                        # Comorbidities
                        comorbidities = chronic.get('Comorbidities', [])
                        if comorbidities:
                            st.write("**Comorbidities:**")
                            for comorbidity in comorbidities:
                                st.write(f"- {comorbidity}")

                    with tabs[3]:
                        # Display treatment plan
                        st.subheader("Treatment and Follow-up Plan")

                        # Therapeutic interventions
                        therapies = patient_record.get('structured_data', {}).get('TherapeuticInterventions', {})

                        # Medications
                        medications = therapies.get('Medications', [])
                        if medications:
                            st.write("**Medications:**")
                            for med in medications:
                                st.write(f"- {med}")

                        # Procedures
                        procedures = therapies.get('Procedures', [])
                        if procedures:
                            st.write("**Procedures:**")
                            for proc in procedures:
                                st.write(f"- {proc}")

                        # Follow-up plan
                        followup = patient_record.get('structured_data', {}).get('Follow-upPlan', {})

                        # Planned consultations
                        consultations = followup.get('PlannedConsultations', [])
                        if consultations:
                            st.write("**Planned Consultations:**")
                            for consult in consultations:
                                st.write(f"- {consult}")

                        # Scheduled tests
                        tests = followup.get('ScheduledTests', [])
                        if tests:
                            st.write("**Scheduled Tests:**")
                            for test in tests:
                                st.write(f"- {test}")

                        # Next appointment
                        appointments = followup.get('NextAppointmentDetails', [])
                        if appointments:
                            st.write("**Next Appointment Details:**")
                            for appt in appointments:
                                st.write(f"- {appt}")

                        # Visit timeline
                        timeline = patient_record.get('structured_data', {}).get('VisitTimeline', [])
                        if timeline:
                            st.write("**Visit Timeline:**")
                            for visit in timeline:
                                st.write(f"- {visit}")

                    with tabs[4]:
                        # Display raw conversation
                        st.text_area("Raw Conversation Text",
                                     patient_record.get('raw_text', 'No conversation text available'), height=400)

                else:
                    st.warning("Patient record not found in the database")

            except Exception as e:
                st.error(f"Error retrieving patient data: {e}")
    else:
        st.warning("No patients match your search criteria. Please adjust your filters.")


def show_browser_page():
    st.header("Patient Record Browser")

    # Clear cache if refresh is needed
    if st.session_state.refresh_data:
        st.cache_data.clear()
        st.session_state.refresh_data = False

    # Load metadata with refresh
    metadata_list = load_metadata(_refresh=True)

    # Create dataframe
    if metadata_list:
        df = pd.DataFrame(metadata_list)

        # Sort options and filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_option = st.selectbox(
                "Sort by:",
                ["patient_id", "Age", "Gender", "MRN", "Diagnosis"]
            )

        with col2:
            filter_gender = st.selectbox(
                "Filter by gender:",
                ["All", "Male", "Female"]
            )

        with col3:
            search_term = st.text_input("Search by MRN or diagnosis:")

        filtered_df = df.copy()

        if filter_gender != "All":
            filtered_df = filtered_df[filtered_df["Gender"] == filter_gender]

        if search_term:
            search_mask = (
                    filtered_df["MRN"].astype(str).str.contains(search_term, case=False, na=False) |
                    filtered_df["Diagnosis"].astype(str).str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[search_mask]

        if sort_option in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=sort_option)

        st.dataframe(filtered_df, use_container_width=True)
        st.subheader("Patient Detail View")
        selected_id = st.selectbox("Select Patient ID", sorted([m['patient_id'] for m in metadata_list]))


        if selected_id:
            patient_data = next((m for m in metadata_list if m['patient_id'] == selected_id), None)

            if patient_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MRN", patient_data.get('MRN', 'N/A'))
                with col2:
                    st.metric("Age", patient_data.get('Age', 'N/A'))
                with col3:
                    st.metric("Gender", patient_data.get('Gender', 'N/A'))

                st.subheader(f"Diagnosis: {patient_data.get('Diagnosis', 'N/A')}")

                try:
                    with open(input_json_path, "r") as f:
                        data = json.load(f)

                    # Find the patient record
                    patient_record = next((item for item in data if item['patient_id'] == selected_id) or item['MRN']== MRN, None)

                    if patient_record:
                        # Create tabs for better organization of data
                        tabs = st.tabs([
                            "Full Patient Record",
                            "Clinical Summary",
                            "Diagnostic Information",
                            "Treatment Plan",
                            "Conversation"
                        ])
                        with tabs[0]:
                            # Display full JSON data
                            st.json(patient_record.get('structured_data', {}))

                        with tabs[1]:
                            # Display clinical summary
                            clinical_summary = patient_record.get('structured_data', {}).get('ClinicalSummary', {})

                            if clinical_summary:
                                # Active symptoms
                                st.subheader("Active Symptoms")
                                symptoms = clinical_summary.get('ActiveSymptoms', [])
                                if symptoms:
                                    for symptom in symptoms:
                                        st.write(f"- {symptom}")
                                else:
                                    st.write("No active symptoms recorded")

                                # Negative findings
                                st.subheader("Negative Findings")
                                neg_findings = clinical_summary.get('NegativeFindings', [])
                                if neg_findings:
                                    for finding in neg_findings:
                                        st.write(f"- {finding}")
                                else:
                                    st.write("No negative findings recorded")

                                # Narrative summary if available
                                narrative = patient_record.get('structured_data', {}).get('SummaryNarrative', {})
                                if narrative:
                                    st.subheader("Clinical Narrative")

                                    # Clinical course
                                    course = narrative.get('ClinicalCourseProgression', '')
                                    if course:
                                        st.write(f"**Clinical Course:** {course}")

                                    # Diagnostic journey
                                    journey = narrative.get('DiagnosticJourney', '')
                                    if journey:
                                        st.write(f"**Diagnostic Journey:** {journey}")

                                    # Treatment response
                                    response = narrative.get('TreatmentResponse', '')
                                    if response:
                                        st.write(f"**Treatment Response:** {response}")

                                    # Ongoing concerns
                                    concerns = narrative.get('OngoingConcerns', '')
                                    if concerns:
                                        st.write(f"**Ongoing Concerns:** {concerns}")
                            else:
                                st.write("No clinical summary available")

                        with tabs[2]:
                            # Display diagnostic information
                            st.subheader("Diagnostic Conclusions")
                            diag_conclusions = patient_record.get('structured_data', {}).get('DiagnosticConclusions',
                                                                                             [])
                            if diag_conclusions:
                                for conclusion in diag_conclusions:
                                    st.write(f"- {conclusion}")
                            else:
                                st.write("No diagnostic conclusions available")

                            # Diagnostic evidence
                            st.subheader("Diagnostic Evidence")
                            diag_evidence = patient_record.get('structured_data', {}).get('DiagnosticEvidence', {})

                            # Imaging findings
                            img_findings = diag_evidence.get('ImagingFindings', [])
                            if img_findings:
                                st.write("**Imaging Findings:**")
                                for finding in img_findings:
                                    st.write(f"- {finding}")

                            # Lab results
                            lab_results = diag_evidence.get('LabResults', [])
                            if lab_results:
                                st.write("**Laboratory Results:**")
                                for result in lab_results:
                                    st.write(f"- {result}")

                            # Pathology findings
                            path_findings = diag_evidence.get('PathologyFindings', [])
                            if path_findings:
                                st.write("**Pathology Findings:**")
                                for finding in path_findings:
                                    st.write(f"- {finding}")

                        with tabs[3]:

                            # Therapeutic interventions
                            st.subheader("Therapeutic Interventions")
                            therapies = patient_record.get('structured_data', {}).get('TherapeuticInterventions', {})

                            # Medications
                            medications = therapies.get('Medications', [])
                            if medications:
                                st.write("**Medications:**")
                                for med in medications:
                                    st.write(f"- {med}")
                            else:
                                st.write("No medications recorded")

                            # Procedures
                            procedures = therapies.get('Procedures', [])
                            if procedures:
                                st.write("**Procedures:**")
                                for proc in procedures:
                                    st.write(f"- {proc}")
                            else:
                                st.write("No procedures recorded")

                            # Follow-up plan
                            st.subheader("Follow-up Plan")
                            followup = patient_record.get('structured_data', {}).get('Follow-upPlan', {})

                            if followup:
                                # Planned consultations
                                consultations = followup.get('PlannedConsultations', [])
                                if consultations:
                                    st.write("**Planned Consultations:**")
                                    for consult in consultations:
                                        st.write(f"- {consult}")

                                # Scheduled tests
                                tests = followup.get('ScheduledTests', [])
                                if tests:
                                    st.write("**Scheduled Tests:**")
                                    for test in tests:
                                        st.write(f"- {test}")

                                # Next appointment
                                appointments = followup.get('NextAppointmentDetails', [])
                                if appointments:
                                    st.write("**Next Appointment Details:**")
                                    for appt in appointments:
                                        st.write(f"- {appt}")
                            else:
                                st.write("No follow-up plan recorded")

                            # Chronic conditions
                            st.subheader("Chronic Conditions")
                            chronic = patient_record.get('structured_data', {}).get('ChronicConditions', {})

                            if chronic:
                                # Chronic diseases
                                diseases = chronic.get('ChronicDiseases', [])
                                if diseases:
                                    st.write("**Chronic Diseases:**")
                                    for disease in diseases:
                                        st.write(f"- {disease}")

                                # Comorbidities
                                comorbidities = chronic.get('Comorbidities', [])
                                if comorbidities:
                                    st.write("**Comorbidities:**")
                                    for comorbidity in comorbidities:
                                        st.write(f"- {comorbidity}")
                            else:
                                st.write("No chronic conditions recorded")

                            # Visit timeline
                            st.subheader("Visit Timeline")
                            timeline = patient_record.get('structured_data', {}).get('VisitTimeline', [])
                            if timeline:
                                for visit in timeline:
                                    st.write(f"- {visit}")
                            else:
                                st.write("No visit timeline recorded")

                        with tabs[4]:
                            # Display original conversation
                            st.text_area("Full Clinical Text", patient_record.get('raw_text', 'No text available'),
                                         height=400)
                    else:
                        st.warning(f"Could not find detailed record for patient {selected_id}")

                except Exception as e:
                    st.error(f"Error loading patient data: {e}")
    else:
        st.error("No patient data available. Please add patient records first.")


def extract_gender(structured_data):
    """Extract gender from structured data"""
    if isinstance(structured_data, dict):
        demographics = structured_data.get('PatientDemographics', {})
        return demographics.get('Gender', 'Unknown')
    return 'Unknown'


def show_analytics_page():
    st.header("HEPABOT Analytics")

    # Load metadata
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Process data for analytics
    genders = []
    ages = []
    diagnoses = []

    for patient in data:
        if 'structured_data' in patient:
            demographics = patient['structured_data'].get('PatientDemographics', {})
            genders.append(demographics.get('Gender', 'Unknown'))

            # Handle age - ensure it's numeric
            age = demographics.get('Age', '')
            try:
                age = int(age)
                ages.append(age)
            except (ValueError, TypeError):
                pass

            diagnoses.append(demographics.get('Diagnosis', 'Unknown'))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Demographics")

        # Gender distribution
        gender_counts = pd.Series(genders).value_counts()
        st.bar_chart(gender_counts)

        # Age distribution
        if ages:
            st.subheader("Age Distribution")
            age_df = pd.DataFrame({'Age': ages})

            # Define your own bins
            min_age = int(min(ages)) // 10 * 10
            max_age = int(max(ages)) // 10 * 10 + 10
            bins = list(range(min_age, max_age + 1, 10))

            # Now cut using these bins
            age_bins = pd.cut(age_df['Age'], bins=bins, right=False)

            hist_values = pd.DataFrame(age_bins.value_counts().sort_index())
            hist_values.index = hist_values.index.map(
                lambda x: f"{int(x.left)}–{int(x.right - 1)}")

            st.bar_chart(hist_values)

    import altair as alt

    with col2:
        st.subheader("Diagnosis Distribution")

        # Count diagnoses
        diagnosis_counts = pd.Series(diagnoses).value_counts().head(10).reset_index()
        diagnosis_counts.columns = ['Diagnosis', 'Count']

        chart = alt.Chart(diagnosis_counts).mark_bar().encode(
            x=alt.X(
                'Diagnosis:N',
                sort='-y',
                axis=alt.Axis(
                    labelAngle=-45,
                    labelFontSize=12,
                    labelOverlap=False
                )
            ),
            y=alt.Y('Count:Q')
        ).properties(
            width=700,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)


def show_generate_report_page():
    OUTPUT_JSON = input_json_path
    output_directory = "patient_records"

    # Check and load vector database if it exists but isn't loaded
    if os.path.exists(DB_PATH) and not st.session_state.get('vector_db_created', False):
        st.session_state.vector_db_created = True
        try:
            vector_db = load_vector_db(
                persist_directory=DB_PATH,
                collection_name=COLLECTION_NAME
            )
            if vector_db:
                st.session_state.rag_chain = create_rag_chain(vector_db, MODEL_NAME)
        except Exception as e:
            st.error(f"Error loading vector database: {e}")

    st.header("🩺 Generate Patient Report")
    st.write("Upload a clinical conversation PDF or audio file to generate a structured medical report")

    # Tabs for upload options
    tab1, tab2 = st.tabs(["Upload Audio", "Upload PDF"])

    with tab2:
        uploaded_pdf = st.file_uploader("Upload a clinical conversation PDF", type=["pdf"], key="pdf_uploader")

    with tab1:
        uploaded_audio = st.file_uploader(
            "Upload an audio file",
            type=["mp3", "wav", "m4a", "flac", "aac", "ogg"],
            key="audio_uploader"
        )

        if uploaded_audio:
            st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")

    if uploaded_pdf or uploaded_audio:
        if uploaded_pdf:
            # Save PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as patient_file:
                patient_file.write(uploaded_pdf.read())
                tmp_path = patient_file.name
        elif uploaded_audio:
            # Process audio to PDF
            with st.spinner("🔊 Transcribing audio..."):
                pdf_path, error = process_audio_to_pdf(uploaded_audio)
                if error:
                    st.error(f"Error transcribing audio: {error}")
                    return
                tmp_path = pdf_path

        with st.spinner("🔍 Analyzing data..."):
            try:
                result = process_pdf(tmp_path)

                if not result:
                    st.error("Failed to process PDF. No data extracted.")
                    return

                # Set patient_id to MRN - Check both structured data and raw text
                demographics = result.get('structured_data', {}).get('PatientDemographics', {})
                mrn = demographics.get('MRN', None)

                # If MRN not found in structured data, try to extract it from raw text
                if not mrn and 'raw_text' in result:
                    raw_text = result.get('raw_text', '')
                    mrn_match = re.search(r'[Mm][Rr][Nn]:?\s*(\d+)', raw_text)
                    if mrn_match:
                        mrn = mrn_match.group(1)
                        demographics['MRN'] = mrn
                        result['structured_data']['PatientDemographics'] = demographics
                        st.info(f"MRN extracted from raw text: {mrn}")

                if mrn:
                    result['patient_id'] = mrn
                else:
                    st.warning("No MRN found in patient data. Using default patient ID.")
                    result['patient_id'] = result.get('patient_id', f"patient_{int(time.time())}")

                if os.path.exists(OUTPUT_JSON):
                    with open(OUTPUT_JSON, "r") as f:
                        existing_data = json.load(f)
                else:
                    existing_data = []

                # Check if  patient already exists
                existing_ids = [item.get('patient_id') for item in existing_data]
                if result['patient_id'] in existing_ids:
                    for i, item in enumerate(existing_data):
                        if item['patient_id'] == result['patient_id']:
                            existing_data[i] = result
                            st.info(f"Updated existing record for patient {result['patient_id']}")
                else:
                    existing_data.append(result)

                with open(OUTPUT_JSON, "w") as f:
                    json.dump(existing_data, f, indent=2)
                os.makedirs(output_directory, exist_ok=True)
                save_patient_records(existing_data, output_directory)

                #Add to vector database
                try:
                    collection = get_vector_db()
                    add_patient_record(collection, result)
                except Exception as e:
                    st.error(f"Error adding to vector database: {e}")

                # Generate and display results
                st.subheader("📋 Generated Patient Report")

                # Generate PDF report and provide download button
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as report_file:
                    generate_pdf_report(result, report_file.name)
                    with open(report_file.name, "rb") as f:
                        st.download_button(
                            label="📥 Download Patient Report PDF",
                            data=f,
                            file_name=f"patient_report_{result['patient_id']}.pdf",
                            mime="application/pdf",
                            key=f"download_pdf_{result['patient_id']}"
                        )

                # Display extracted data
                st.subheader("Extracted Patient Data")
                tabs = st.tabs(["Full Report", "Summary"])

                with tabs[1]:
                    demographics = result.get('structured_data', {}).get('PatientDemographics', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MRN", demographics.get('MRN', 'N/A'))
                    with col2:
                        st.metric("Age", demographics.get('Age', 'N/A'))
                    with col3:
                        st.metric("Gender", demographics.get('Gender', 'N/A'))
                    st.write(f"**Diagnosis:** {demographics.get('Diagnosis', 'N/A')}")
                    st.write(
                        f"**Summary:** {result.get('structured_data', {}).get('SummaryNarrative', {}).get('ClinicalCourseProgression', 'N/A')}")

                with tabs[0]:
                    st.json(result.get('structured_data', {}))

                # Generate diagnosis using RAG pipeline
                if st.session_state.get('rag_chain', None):
                    with st.spinner("🩺 Generating AI-assisted diagnosis..."):
                        try:
                            demographics = result.get('structured_data', {}).get('PatientDemographics', {})
                            age = demographics.get('Age', 'unknown')
                            gender = demographics.get('Gender', 'unknown')

                            summary_report = result.get('structured_data', {}).get('SummaryNarrative', {}).get(
                                'ClinicalCourseProgression', 'no symptoms provided')

                            diagnostic_evidence = result.get('structured_data', {}).get('DiagnosticEvidence', {})
                            diagnostic_evidence_str = str(
                                diagnostic_evidence) if diagnostic_evidence else "No diagnostic evidence available"

                            clinical_summary = result.get('structured_data', {}).get('ClinicalSummary', {})
                            clinical_summary_str = str(
                                clinical_summary) if clinical_summary else "No clinical summary available"

                            question = (
                                f"Patient is {age} years old, {gender}, showing symptoms: {clinical_summary_str}, his Lab test include"
                                f" {diagnostic_evidence_str}, and his overall summary report is {summary_report} What is the likely diagnosis?")

                            # Get diagnosis from RAG chain
                            diagnosis = ask_question(st.session_state.rag_chain, question)

                            # Display diagnosis
                            st.subheader("AI-Assisted Diagnosis")
                            st.markdown(diagnosis)

                            # Download diagnosis
                            col1, _ = st.columns(2)
                            with col1:
                                if st.button("Download Diagnosis"):
                                    download_data = diagnosis.encode()
                                    st.markdown(
                                        get_download_link(
                                            download_data,
                                            f"diagnosis_{result['patient_id']}.txt",
                                            "Download Diagnosis Result"
                                        ),
                                        unsafe_allow_html=True
                                    )
                        except Exception as e:
                            st.error(f"Error generating diagnosis: {str(e)}")
                else:
                    st.warning("No vector database or RAG chain available to generate diagnosis.")

                # Refresh metadata cache
                st.session_state.refresh_data = True
                st.cache_data.clear()

                if st.button("View in Patient Browser"):
                    st.session_state.page = "browser"
                    st.experimental_rerun()

            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

def generate_pdf_report(patient_data, output_path):
    """Generateate a PDF report for the patient data."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y_position = height - 50
    line_height = 14

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, f"Patient Report - ID: {patient_data.get('patient_id', 'Unknown')}")
    y_position -= 30

    # Demographics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Patient Demographics")
    y_position -= line_height
    c.setFont("Helvetica", 10)
    demographics = patient_data.get('structured_data', {}).get('PatientDemographics', {})
    for key, value in demographics.items():
        c.drawString(60, y_position, f"{key}: {value}")
        y_position -= line_height
    y_position -= 10

    # Clinical Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Clinical Summary")
    y_position -= line_height
    c.setFont("Helvetica", 10)
    clinical_summary = patient_data.get('structured_data', {}).get('ClinicalSummary', {})
    symptoms = clinical_summary.get('ActiveSymptoms', [])
    if symptoms:
        c.drawString(60, y_position, "Active Symptoms:")
        y_position -= line_height
        for symptom in symptoms:
            c.drawString(70, y_position, f"- {symptom}")
            y_position -= line_height
    neg_findings = clinical_summary.get('NegativeFindings', [])
    if neg_findings:
        c.drawString(60, y_position, "Negative Findings:")
        y_position -= line_height
        for finding in neg_findings:
            c.drawString(70, y_position, f"- {finding}")
            y_position -= line_height
    y_position -= 10

    # Diagnostic Information
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Diagnostic Information")
    y_position -= line_height
    c.setFont("Helvetica", 10)
    diag_conclusions = patient_data.get('structured_data', {}).get('DiagnosticConclusions', [])
    if diag_conclusions:
        c.drawString(60, y_position, "Diagnostic Conclusions:")
        y_position -= line_height
        for conclusion in diag_conclusions:
            c.drawString(70, y_position, f"- {conclusion}")
            y_position -= line_height
    y_position -= 10

    # Treatment Plan
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Treatment Plan")
    y_position -= line_height
    c.setFont("Helvetica", 10)
    therapies = patient_data.get('structured_data', {}).get('TherapeuticInterventions', {})
    medications = therapies.get('Medications', [])
    if medications:
        c.drawString(60, y_position, "Medications:")
        y_position -= line_height
        for med in medications:
            c.drawString(70, y_position, f"- {med}")
            y_position -= line_height
    procedures = therapies.get('Procedures', [])
    if procedures:
        c.drawString(60, y_position, "Procedures:")
        y_position -= line_height
        for proc in procedures:
            c.drawString(70, y_position, f"- {proc}")
            y_position -= line_height

    c.showPage()
    c.save()


if __name__ == "__main__":
    main()