from workflow import (
    SUPPORTED_MANUFACTURERS,
    SUPPORTED_MODELS,
    clear_knowledge_base,
    get_kb_stats,
    ingest_uploaded_files,
    ingest_backend_google_drive,
    ingest_google_drive_link,
    troubleshoot,
)

st.set_page_config(
    page_title="Marine AI Troubleshooting Agent",
    page_icon="⚓",
    layout="wide",
)

st.title("⚓ Marine AI Troubleshooting Agent")
st.caption("AI-powered marine engine diagnostics using OEM manuals, RAG, and technical research")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb" not in st.session_state:
    st.session_state.kb = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.divider()

st.subheader("Backend OEM Manuals")

if st.button(
    "📚 Load / Refresh Backend Manuals",
    use_container_width=True,
):
    try:
        with st.spinner(
            "Loading OEM manuals from Google Drive..."
        ):
            st.session_state.kb = ingest_backend_google_drive(
                st.session_state.kb
            )

        st.success("Backend OEM manuals loaded.")
        st.rerun()

    except Exception as exc:
        st.error(
            f"Backend Google Drive loading failed: {exc}"
        )
    st.header("System Status")
    try:
        from workflow import get_api_status
        status = get_api_status()
        st.write("Groq API:", "✅ Configured" if status["groq"] else "❌ Missing")
        st.write("Online research:", "✅ Configured" if status["tavily"] else "⚪ Not configured")
    except Exception as exc:
        st.error(f"Status error: {exc}")

    stats = get_kb_stats(st.session_state.kb)
    st.write(f"Documents indexed: **{stats['documents']}**")
    st.write(f"Chunks indexed: **{stats['chunks']}**")

    search_mode = st.radio(
        "Search mode",
        ["Uploaded Manuals", "Online Research", "Both"],
        index=0,
    )
    st.session_state.search_mode = search_mode

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    if st.button("Clear knowledge base", use_container_width=True):
        st.session_state.kb = clear_knowledge_base()
        st.rerun()

    st.divider()
    st.warning(
        "Safety: This assistant supports troubleshooting and maintenance decisions. "
        "It does not replace the applicable OEM manual, authorized technician, "
        "class requirements, or vessel safety procedures."
    )

st.subheader("A. Engine Identification")

col1, col2 = st.columns(2)
with col1:
    manufacturer = st.selectbox(
        "Manufacturer",
        SUPPORTED_MANUFACTURERS,
        index=0,
    )
    model_options = ["Enter model manually"] + SUPPORTED_MODELS
    model_choice = st.selectbox("Engine model", model_options)
    if model_choice == "Enter model manually":
        engine_model = st.text_input("Exact engine model", placeholder="e.g. CAT C18")
    else:
        engine_model = model_choice

    serial_number = st.text_input(
        "Serial number (optional)",
        placeholder="Enter exact engine serial number if available",
    )
    vessel_name = st.text_input("Vessel / equipment name (optional)")

with col2:
    operating_hours = st.text_input(
        "Operating hours (optional)",
        placeholder="e.g. 4,250 h",
    )
    application_type = st.selectbox(
        "Application",
        ["Main propulsion", "Auxiliary", "Generator", "Outboard", "Other"],
    )

st.subheader("B. Defect Description")
defect = st.text_area(
    "Describe the defect / alarm / operating condition",
    height=180,
    placeholder=(
        "Example: Coolant temperature rises during high-load operation. "
        "After approximately 20 minutes an alarm appears. Describe any "
        "smoke, noise, vibration, pressure, RPM, fluid leakage, recent work, "
        "or alarm code if known."
    ),
)

st.subheader("C. Manual Upload / Knowledge Base")
uploaded = st.file_uploader(
    "Upload OEM manuals, service manuals, wiring diagrams, parts catalogs, PDF/DOCX/TXT/MD",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if st.button("Index uploaded manuals", type="secondary"):
    if not uploaded:
        st.info("Select one or more manuals first.")
    else:
        try:
            with st.spinner("Extracting, chunking, embedding and indexing manuals..."):
                st.session_state.kb = ingest_uploaded_files(
                    uploaded,
                    st.session_state.kb,
                    manufacturer=manufacturer,
                    engine_model=engine_model,
                )
            st.success("Manual indexing completed.")
            st.rerun()
        except Exception as exc:
            st.error(f"Manual indexing failed: {exc}")

drive_url = st.text_input(
    "Google Drive share link (optional)",
    placeholder="Paste a shareable Google Drive file link",
)
if st.button("Index Google Drive manual", type="secondary"):
    if not drive_url.strip():
        st.info("Paste a Google Drive file link first.")
    else:
        try:
            with st.spinner("Downloading and indexing Google Drive document..."):
                st.session_state.kb = ingest_google_drive_link(
                    drive_url,
                    st.session_state.kb,
                    manufacturer=manufacturer,
                    engine_model=engine_model,
                )
            st.success("Google Drive document indexed.")
            st.rerun()
        except Exception as exc:
            st.error(f"Google Drive ingestion failed: {exc}")

st.divider()
if st.button("🔧 Diagnose / Troubleshoot", type="primary", use_container_width=True):
    if not manufacturer or not engine_model.strip() or not defect.strip():
        st.error("Manufacturer, exact engine model and defect description are required.")
    else:
        context = {
            "manufacturer": manufacturer,
            "engine_model": engine_model.strip(),
            "serial_number": serial_number.strip(),
            "vessel_name": vessel_name.strip(),
            "operating_hours": operating_hours.strip(),
            "application_type": application_type,
        }
        try:
            with st.spinner("Retrieving evidence and preparing technical analysis..."):
                result = troubleshoot(
                    context,
                    defect.strip(),
                    st.session_state.kb,
                    search_mode=st.session_state.get("search_mode", "Uploaded Manuals"),
                )
            st.session_state.last_result = result
            st.session_state.messages.append(
                {"role": "user", "content": defect.strip()}
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"]}
            )
        except Exception as exc:
            st.error(f"Troubleshooting failed: {exc}")

result = st.session_state.last_result
if result:
    st.subheader("Troubleshooting Result")

    if result.get("web_used"):
        st.info("Online research was actually executed and returned results.")
    else:
        st.info("No online-search results were used for this response.")

    st.markdown(result["answer"])

    with st.expander("Retrieved evidence", expanded=False):
        evidence = result.get("evidence", [])
        if not evidence:
            st.write("No relevant evidence was retrieved.")
        for i, item in enumerate(evidence, 1):
            st.markdown(
                f"**{i}. {item.get('title', 'Untitled')}**  \n"
                f"Source: {item.get('source_type', 'Unknown')}  \n"
                f"Model: {item.get('engine_model', 'Not specified')}  \n"
                f"Page: {item.get('page', 'N/A')}"
            )
            st.caption(item.get("text", "")[:3000])

    with st.expander("Raw research results", expanded=False):
        for item in result.get("web_results", []):
            st.markdown(
                f"**{item.get('title', 'Untitled')}**  \n"
                f"{item.get('url', '')}"
            )
            st.caption(item.get("content", "")[:3000])

st.subheader("Follow-up Questions")
for message in st.session_state.messages[-6:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

follow_up = st.chat_input(
    "Ask a follow-up using the currently indexed manuals, e.g. 'What should I check next?'"
)
if follow_up:
    context = {
        "manufacturer": manufacturer,
        "engine_model": engine_model.strip(),
        "serial_number": serial_number.strip(),
        "vessel_name": vessel_name.strip(),
        "operating_hours": operating_hours.strip(),
        "application_type": application_type,
    }
    try:
        with st.spinner("Retrieving evidence for follow-up..."):
            result = troubleshoot(
                context,
                follow_up,
                st.session_state.kb,
                search_mode=st.session_state.get("search_mode", "Uploaded Manuals"),
            )
        st.session_state.last_result = result
        st.session_state.messages.append({"role": "user", "content": follow_up})
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
        st.rerun()
    except Exception as exc:
        st.error(f"Follow-up failed: {exc}")

st.caption(
    "Supported starting models include MTU 10V 2000 M94, MTU 12V 2000 M94, "
    "MTU 12V 2000 M96L, MTU 16V 4000 M90, MAN 12V 175D and MAN 16V 175D. "
    "The architecture also accepts other manufacturers/models."
)
