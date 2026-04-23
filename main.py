import streamlit as st
from streamlit_option_menu import option_menu
import importlib

st.set_page_config(page_title="Walkability Analysis", layout="wide")

# Page registry — modules loaded lazily when user navigates to them
PAGES = {
    "Introduction":  ("pages.introduction", "info-circle"),
    "Data Prep/EDA": ("pages.data_prep",    "database"),
    "Models":        ("pages.models",        "bar-chart"),
    "Conclusion":    ("pages.conclusion",    "check-circle"),
    "About Me":      ("pages.about_me",      "person"),
}


def main():
    with st.sidebar:
        selected = option_menu(
            "Walkability",
            list(PAGES.keys()),
            icons=[v[1] for v in PAGES.values()],
            menu_icon="cast",
            default_index=0,
        )
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Analysis of walkability in urban areas.")
        st.markdown("Developed by SEJAL HUKARE.")

    module_path = PAGES[selected][0]
    page = importlib.import_module(module_path)
    page.app()


if __name__ == "__main__":
    main()