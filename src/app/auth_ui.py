"""
Streamlit Authentication & Researcher Profile UI Component.
Provides sign-in, registration, demo account quick-fill, and session state management.
"""

from typing import Any, Dict, Optional
import streamlit as st
from src.auth.service import auth_service


def get_current_user() -> Optional[Dict[str, Any]]:
    """Returns the authenticated user dict from session state if logged in."""
    if st.session_state.get("is_authenticated", False):
        return st.session_state.get("authenticated_user")
    return None


def render_auth_sidebar() -> None:
    """Renders authentication controls, profile badges, and sign-in modal in Streamlit sidebar."""
    # Initialize session state keys
    if "is_authenticated" not in st.session_state:
        st.session_state["is_authenticated"] = False
        st.session_state["authenticated_user"] = None

    st.sidebar.markdown("---")

    if st.session_state["is_authenticated"]:
        user = st.session_state["authenticated_user"]
        st.sidebar.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 10px; margin-bottom: 8px;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #38BDF8;">👤 Researcher Active Session</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin-top: 2px;">{user.get('full_name', user['username'])}</div>
            <div style="font-size: 0.75rem; color: #94A3B8;">Role: <b style="color: #CBD5E1;">{user.get('role', 'Researcher')}</b></div>
            <div style="font-size: 0.75rem; color: #64748B;">{user.get('institution', 'Research Consortium')}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.sidebar.button("🚪 Sign Out", key="auth_signout_btn", use_container_width=True):
            st.session_state["is_authenticated"] = False
            st.session_state["authenticated_user"] = None
            st.toast("Signed out successfully.")
            st.rerun()
    else:
        with st.sidebar.expander("🔑 Researcher Sign-In & Access", expanded=False):
            tab_login, tab_reg = st.tabs(["Sign In", "Register"])

            with tab_login:
                st.caption("Sign in to save private screening libraries and audit history:")

                # 1-Click Fast Demo Login
                if st.button("⚡ 1-Click Demo Login (Lead PI)", key="quick_admin_login", type="primary", use_container_width=True):
                    ok, msg, user_obj = auth_service.authenticate("admin", "adminpassword123")
                    if ok and user_obj:
                        st.session_state["is_authenticated"] = True
                        st.session_state["authenticated_user"] = user_obj
                        st.rerun()

                st.markdown("<div style='text-align: center; font-size: 0.75rem; color: #64748B; margin: 6px 0;'>— OR MANUAL LOGIN —</div>", unsafe_allow_html=True)

                demo_pick = st.selectbox(
                    "Autofill Demo Credentials:",
                    options=["-- Custom Credentials --", "Lead PI (admin)", "Scientist (scientist)", "Reviewer (reviewer)"],
                    key="demo_pick_select"
                )

                default_u = ""
                default_p = ""
                if demo_pick == "Lead PI (admin)":
                    default_u = "admin"
                    default_p = "adminpassword123"
                elif demo_pick == "Scientist (scientist)":
                    default_u = "scientist"
                    default_p = "amr2026pass"
                elif demo_pick == "Reviewer (reviewer)":
                    default_u = "reviewer"
                    default_p = "reviewamr2026"

                user_input = st.text_input(
                    "Username or Email",
                    value=default_u,
                    key="login_user_field"
                )
                pwd_input = st.text_input(
                    "Password",
                    value=default_p,
                    type="password",
                    key="login_pwd_field"
                )

                if st.button("Sign In to Platform", key="login_submit_btn", use_container_width=True):
                    final_user = user_input.strip() or default_u
                    final_pwd = pwd_input.strip() or default_p
                    if not final_user or not final_pwd:
                        st.error("Please enter both username and password.")
                    else:
                        ok, msg, user_obj = auth_service.authenticate(final_user, final_pwd)
                        if ok and user_obj:
                            st.session_state["is_authenticated"] = True
                            st.session_state["authenticated_user"] = user_obj
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

            with tab_reg:
                st.caption("Create a new researcher account:")
                r_name = st.text_input("Full Name", placeholder="Dr. Jane Doe", key="reg_name")
                r_user = st.text_input("Username", placeholder="janedoe", key="reg_user")
                r_email = st.text_input("Institutional Email", placeholder="j.doe@university.edu", key="reg_email")
                r_inst = st.text_input("Institution / Lab", placeholder="Biomedical Research Center", key="reg_inst")
                r_role = st.selectbox("Role", ["Senior Researcher", "Bioinformatician", "Postdoctoral Fellow", "Reviewer / Auditor"], key="reg_role")
                r_pwd = st.text_input("Password (min 8 chars)", type="password", key="reg_pwd")

                if st.button("Register New Account", key="reg_submit_btn", use_container_width=True):
                    ok, msg, new_user = auth_service.register_user(
                        username=r_user,
                        email=r_email,
                        password=r_pwd,
                        full_name=r_name,
                        role=r_role,
                        institution=r_inst
                    )
                    if ok and new_user:
                        st.session_state["is_authenticated"] = True
                        st.session_state["authenticated_user"] = new_user
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
