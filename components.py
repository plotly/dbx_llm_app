import dash_mantine_components as dmc
import dash
from dash import dcc, html
from dash_iconify import DashIconify
import dash_bootstrap_components as dbc


from dash import html, dcc
import dash_mantine_components as dmc


# the style arguments for the sidebar. We use position:fixed and a fixed width

TOP_NAVBAR = dmc.Navbar(
    className="top-navbar",
    
)


LEFT_SIDEBAR = dmc.Navbar(
    className="sidebar",
    # style={
    #     "backgroundColor": "#0F1D22",
    # },
    mt=20,
    mb=20,
    ml=20,
    children=[
        html.A(
            [
                html.Img(
                src=dash.get_asset_url("plotly_DO.png"),
    style={
        "height": "100%",
        "width": "100%",
        "float": "center",
        "position": "relative",
        "padding-top": 20,
        "padding-right": 20,
        "padding-left": 20,
        "padding-bottom": 10,
    },
),
            ],
            href="https://databricks-dash.aws.plotly.host/databrickslakeside/dbx-console",
        ),
        dmc.Space(h=10),
        dmc.NavLink(
            label="Optimizer",
            icon=DashIconify(icon="tabler:file-delta", width=20, color="#FFFFFF"),
            childrenOffset=20,
            children=[
                dmc.NavLink(
                    label="Build Strategy",
                    href="/delta-optimizer/build-strategy",
                    variant="subtle",
                    icon=DashIconify(icon="mdi:brain", width=20, color="#FFFFFF"),
                    className="nav-link-component",
                ),
                dmc.NavLink(
                    label="Schedule + Run",
                    href="/delta-optimizer/optimizer-runner",
                    variant="subtle",
                    icon=DashIconify(icon="carbon:run", width=20, color="#FFFFFF"),
                    className="nav-link-component",
                ),
                dmc.NavLink(
                    label="Results",
                    href="/delta-optimizer/optimizer-results",
                    variant="subtle",
                    icon=DashIconify(
                        icon="mingcute:presentation-2-fill", width=20, color="#FFFFFF"
                    ),
                    className="nav-link-component",
                ),
            ],
            className="nav-link-component",
        ),
        dmc.NavLink(
            label="Settings",
            href="/delta-optimizer/connection_settings",
            icon=DashIconify(
                icon="material-symbols:settings", width=20, color="#FFFFFF"
            ),
            className="nav-link-component",
        ),
    ],
)


def notification_user(text):
    return dmc.Notification(
        id="notify-user",
        title="Activation Status",
        message=[text],
        disallowClose=False,
        radius="xl",
        icon=[DashIconify(icon="mdi:account-check", width=128)],
        action="show",
        autoClose=False,
    )


def notification_job1_error(text):
    return dmc.Notification(
        id="notify-user",
        title="Activation Status",
        message=[text],
        disallowClose=False,
        radius="xl",
        color="red",
        icon=[DashIconify(icon="material-symbols:error-outline", width=128)],
        action="show",
        autoClose=False,
    )


def notification_delete(text):
    return dmc.Notification(
        id="notify-user",
        title="Deletion Status",
        message=[text],
        disallowClose=False,
        radius="xl",
        color="white",
        icon=[DashIconify(icon="typcn:delete-outline", width=128)],
        action="show",
        autoClose=False,
    )


def notification_update_schedule(text):
    return dmc.Notification(
        id="notify-user",
        title="Schedule Update Status",
        message=[text],
        disallowClose=False,
        radius="xl",
        color="black",
        icon=[DashIconify(icon="line-md:calendar", width=128)],
        action="show",
        autoClose=False,
    )


def notification_update_pause(text):
    return dmc.Notification(
        id="notify-user",
        title="Pause Status",
        message=[text],
        disallowClose=False,
        radius="xl",
        color="black",
        icon=[DashIconify(icon="zondicons:pause-outline", width=128)],
        action="show",
        autoClose=False,
    )


def notification_user_step_1(text):
    return dmc.Notification(
        id="notify-user-step-1",
        title="Job Status",
        message=[text],
        disallowClose=False,
        radius="xl",
        icon=[DashIconify(icon="material-symbols:build-circle-outline", width=128)],
        action="show",
        autoClose=False,
    )


def cluster_loading(text):
    return dmc.Notification(
        id="cluster-loading",
        title="Process initiated",
        message=[text],
        loading=True,
        radius="xl",
        color="orange",
        action="show",
        autoClose=False,
        disallowClose=False,
    )


def cluster_loaded(text):
    return dmc.Notification(
        id="cluster-loaded",
        title="Data loaded",
        message=[text],
        radius="xl",
        color="green",
        action="show",
        icon=DashIconify(icon="akar-icons:circle-check"),
    )


FOOTER_FIXED = dmc.Footer(
    height=50,
    fixed=True,
    className="footer",
    children=[
        html.Div(
            className="footer-content",
            children=[
                html.Div(
                    className="footer-content-item",
                    children=[
                        html.A(
                            "© 2023 Plotly Inc.",
                            href="https://plotly.com/",
                            target="_blank",
                        )
                    ],
                ),
                html.Div(className="footer-content-spacing"),
                html.Div(
                    className="footer-links",
                    children=[
                        html.A(
                            "About",
                            href="https://www.databricks.com/company/about-us",
                            target="_blank",
                        ),
                        html.A(
                            "Databricks+Dash",
                            href="https://dash-demo.plotly.host/plotly-dash-500/snapshot-1684467228-670d42dd",
                            target="_blank",
                        ),
                        html.A(
                            "Blog Posts",
                            href="https://medium.com/plotly/build-real-time-production-data-apps-with-databricks-plotly-dash-269cb64b7575",
                            target="_blank",
                        ),
                        html.A(
                            "Contact",
                            href="https://www.databricks.com/company/contact",
                            target="_blank",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "22rem",
    "padding": "1rem 1rem",
    "background-color": "#f8f9fa",
}


submenu_1 = [
    html.Li(
        # use Row and Col components to position the chevrons
        dbc.Row(
            [
                dbc.Col("Menu 1"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right me-3"),
                    width="auto",
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-1",
    ),
    # we use the Collapse component to hide and reveal the navigation links
    dbc.Collapse(
        [
            dbc.NavLink(
                "Stategy Builder",
                href="/build-strategy",
            ),
            dbc.NavLink("Schedule+Run", href="/optimizer-runner"),
            dbc.NavLink("Results", href="/optimizer-results"),
        ],
        id="submenu-1-collapse",
    ),
]


class GitSource:
    def __init__(self, git_url, git_provider, git_branch):
        self.url = git_url
        self.provider = git_provider
        self.branch = git_branch

    def as_dict(self):
        return {
            "git_url": self.url,
            "git_provider": self.provider,
            "git_branch": self.branch,
        }


class Schedule:
    def __init__(self, quartz_cron_expression, timezone_id, pause_status):
        self.quartz_cron_expression = quartz_cron_expression
        self.timezone_id = timezone_id
        self.pause_status = pause_status

    def as_dict(self):
        return {
            "quartz_cron_expression": self.quartz_cron_expression,
            "timezone_id": self.timezone_id,
            "pause_status": self.pause_status,
        }


class Library:
    def __init__(self, whl_path):
        self.whl = whl_path

    def as_dict(self):
        return {"whl": self.whl}


class NewCluster:
    def __init__(
        self,
        node_type_id,
        spark_version,
        num_workers,
        spark_conf,
        spark_env_vars,
        enable_elastic_disk,
    ):
        self.node_type_id = node_type_id
        self.spark_version = spark_version
        self.num_workers = num_workers
        self.spark_conf = spark_conf
        self.spark_env_vars = spark_env_vars
        self.enable_elastic_disk = enable_elastic_disk

    def as_dict(self):
        return {
            "node_type_id": self.node_type_id,
            "spark_version": self.spark_version,
            "num_workers": self.num_workers,
            "spark_conf": self.spark_conf,
            "spark_env_vars": self.spark_env_vars,
            "enable_elastic_disk": self.enable_elastic_disk,
        }


class NotebookTask:
    def __init__(self, notebook_path, base_parameters):
        self.notebook_path = notebook_path
        self.base_parameters = base_parameters

    def as_dict(self):
        return {
            "notebook_path": self.notebook_path,
            "base_parameters": self.base_parameters,
        }