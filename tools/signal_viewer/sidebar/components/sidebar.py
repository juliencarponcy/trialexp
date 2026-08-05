"""Sidebar component for the app."""

from sidebar import styles

import reflex as rx

from trialexp.process.folder_org.utils import build_session_info_cohort


def sidebar_header() -> rx.Component:
    """Sidebar header.

    Returns:
        The sidebar header component.
    """
    return rx.hstack(
        # The logo.
        rx.color_mode_cond(
            rx.image(src="/reflex_black.svg", height="2em"),
            rx.image(src="/reflex_white.svg", height="2em"),
        ),
        rx.spacer(),
        rx.link(
            rx.button(
                rx.icon("github"),
                color_scheme="gray",
                variant="soft",
            ),
            href="https://github.com/reflex-dev/reflex",
        ),
        align="center",
        width="100%",
        border_bottom=styles.border,
        padding_x="1em",
        padding_y="2em",
    )


def sidebar_footer() -> rx.Component:
    """Sidebar footer.

    Returns:
        The sidebar footer component.
    """
    return rx.hstack(
        rx.spacer(),
        rx.link(
            rx.text("Docs"),
            href="https://reflex.dev/docs/getting-started/introduction/",
            color_scheme="gray",
        ),
        rx.link(
            rx.text("Blog"),
            href="https://reflex.dev/blog/",
            color_scheme="gray",
        ),
        width="100%",
        border_top=styles.border,
        padding="1em",
    )


def sidebar_item(text: str, url: str) -> rx.Component:
    """Sidebar item.

    Args:
        text: The text of the item.
        url: The URL of the item.

    Returns:
        rx.Component: The sidebar item component.
    """
    # Whether the item is active.
    active = (rx.State.router.page.path == url.lower()) | (
        (rx.State.router.page.path == "/") & text == "Home"
    )

    return rx.link(
        rx.hstack(
            rx.text(
                text,
            ),
            bg=rx.cond(
                active,
                rx.color("accent", 2),
                "transparent",
            ),
            border=rx.cond(
                active,
                f"1px solid {rx.color('accent', 6)}",
                f"1px solid {rx.color('gray', 6)}",
            ),
            color=rx.cond(
                active,
                styles.accent_text_color,
                styles.text_color,
            ),
            align="center",
            border_radius=styles.border_radius,
            width="100%",
            padding="1em",
        ),
        href=url,
        width="100%",
    )


class SessionSelectState(rx.State):
    """The state for the session select component."""
    
    root_path = '/mnt/Magill_Lab/Julien/ASAP/Data'
    df_session_info = build_session_info_cohort(root_path)
    cohorts = df_session_info.cohort.unique().tolist()

    cohort_list: list[str] = cohorts
    animal_id_list: list[str] = ['undefined']
    session_id_list: list[str] = ['undefined']
    
    cohort: str = ""
    animal_id: str = ""
    session_id: str = ""
    
    @rx.event
    def set_cohort(self, cohort: str):
        self.cohort = cohort
        idx = (self.df_session_info.cohort==self.cohort)
        self.animal_id_list = self.df_session_info[idx].animal_id.unique().tolist()

    @rx.event
    def set_animal_id(self, animal_id: str):
        self.animal_id = animal_id
        idx = (self.df_session_info.animal_id==self.animal_id)
        self.session_id_list = self.df_session_info[idx].session_id.unique().tolist()

    @rx.event
    def set_session_id(self, session_id: str):
        self.session_id = session_id



def sidebar() -> rx.Component:
    """The sidebar.

    Returns:
        The sidebar component.
    """
    # Get all the decorated pages and add them to the sidebar.
    from reflex.page import get_decorated_pages

    return rx.box(
        rx.vstack(
            sidebar_header(),
            rx.vstack(
                rx.select(
                    SessionSelectState.cohort_list,
                    label="Cohort",
                    value=SessionSelectState.cohort,
                    on_change=SessionSelectState.set_cohort,
                ),
                rx.select(
                    SessionSelectState.animal_id_list,
                    label="Animal ID",
                    on_change=SessionSelectState.set_animal_id,
                ),
                rx.select(
                    SessionSelectState.session_id_list,
                    label="Session ID",
                    on_change=SessionSelectState.set_session_id,
                ),
                width="100%",
                overflow_y="auto",
                align_items="flex-start",
                padding="1em",
            ),           
            rx.button("Plot", color="accent", margin="1em", width="80%"),
            rx.spacer(),
            sidebar_footer(),
            height="100dvh",
        ),
        display=["none", "none", "block"],
        min_width=styles.sidebar_width,
        height="100%",
        position="sticky",
        top="0px",
        border_right=styles.border,
    )
