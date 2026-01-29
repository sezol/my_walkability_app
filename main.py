import streamlit as st
from streamlit_option_menu import option_menu
import walkability_app.introduction as introduction, data_prep, models, conclusion, about_me


st.set_page_config(page_title="Walkability Analysis", layout="wide")



class MainApp:
    def __init__(self):
        self.pages = {
            "Introduction": introduction,
            "Data Prep/EDA": data_prep,
            "Models": models,
            "Conclusion": conclusion,
            "About Me": about_me, 
        }
    
    def run(self):
        with st.sidebar:
            selected_page = option_menu(
                "Walkability",
                list(self.pages.keys()),
                icons=["info-circle", "database", "bar-chart", "check-circle"],
                menu_icon="cast",
                default_index=0,
            )
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("This app provides an analysis of walkability in urban areas.")
            st.markdown("Developed by SEJAL HUKARE.")
        
        # Just call the page once
        page_module = self.pages[selected_page]
        page_module.app()

# Create instance and run (OUTSIDE the class)
if __name__ == "__main__":
    app = MainApp()
    app.run()