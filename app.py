import streamlit as st
import os
import json
import simpy
import pandas as pd
from collections import defaultdict
from io import BytesIO
import zipfile
import matplotlib.pyplot as plt
from graphviz import Digraph
import random  # Added for MTBF/MTTR distributions
import numpy as np  # Used for plotting distribution curves

# ========== Configuration ==========
SAVE_DIR = "simulations"
USERNAME = "aksh.fii"
PASSWORD = "foxy123"
os.makedirs(SAVE_DIR, exist_ok=True)
st.set_page_config(page_title="Production Line Simulator", layout="wide")

# ========== Session State Setup ==========
for key in ["authenticated", "page", "simulation_data", "group_names", "connections", "from_stations"]:
    if key not in st.session_state:
        if key == "authenticated":
            st.session_state[key] = False
        elif key == "page":
            st.session_state[key] = "login"
        elif key in ["connections", "from_stations"]:
            st.session_state[key] = {}
        elif key == "group_names":
            st.session_state[key] = []
        else:
            st.session_state[key] = None

# ========== Utility Functions ==========
def determine_lockout_zones(group_names):
    """
    Determine lockout zones based on the presence of 'STACKER' or 'WIP CART' in the group names.
    Each zone includes groups up to and including the next stacker or WIP cart. Remaining groups
    form the final zone. Returns a list of lists where each inner list contains group names
    belonging to a lockout zone.
    """
    zones = []
    current_zone = []
    for g in group_names:
        if not g:
            continue
        current_zone.append(g)
        name_upper = g.upper()
        # Identify boundaries based on keywords
        if ("STACKER" in name_upper) or ("WIP CART" in name_upper) or ("WIPCART" in name_upper):
            zones.append(list(current_zone))
            current_zone = []
    if current_zone:
        zones.append(list(current_zone))
    return zones

# ========== Pages ==========

def login_page():
    st.title("🛠️ Production Line Simulation App (Discrete Event Simulation)")
    st.subheader("🔐 Login")
    user = st.text_input("User ID")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.authenticated = True
            st.session_state.page = "main"
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")

def main_page():
    st.title("🛠️ Production Line Simulation App (Discrete Event Simulation)")
    st.subheader("📊 Simulation Portal")
    st.write("Choose an option:")
    col1, col2 = st.columns(2)
    if col1.button("➕ New Simulation"):
        st.session_state.page = "new"
    if col2.button("📂 Open Simulation"):
        st.session_state.page = "open"

def new_simulation():
    st.title("🛠️ Production Line Simulation App (Discrete Event Simulation)")
    st.subheader("➕ New Simulation Setup")

    col1, col2 = st.columns(2)
    if col1.button("🔙 Back"):
        st.session_state.page = "main"
        return
    if col2.button("🏠 Home"):
        st.session_state.page = "main"
        return

    method = st.radio("How do you want to input your simulation setup?", ["Enter Manually", "Upload Sheet"])

    valid_groups = {}
    group_names = []
    upload_mode = False  # Flag to track if upload mode is active
    # Dictionary to hold reliability metrics per group (mtbf, mttr) when using uploaded sheet
    group_metrics = {}

    if method == "Upload Sheet":
        uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                df.columns = df.columns.str.lower()

                required_columns = {"serial number", "stations", "number of equipment", "cycle time"}
                if not required_columns.issubset(df.columns):
                    st.error("Missing one or more required columns: serial number, stations, number of equipment, cycle time")
                    return

                # Clear previous manual inputs
                st.session_state.group_names = []
                st.session_state.connections = {}
                st.session_state.from_stations = {}

                # Parse stations from the sheet
                for _, row in df.iterrows():
                    station = str(row['stations']).strip().upper()
                    num_eq = int(row['number of equipment'])
                    cycle_times = [float(ct.strip()) for ct in str(row['cycle time']).split(',')]

                    if len(cycle_times) != num_eq:
                        st.warning(f"Station {station}: Number of equipment and cycle time count mismatch.")
                        continue

                    valid_groups[station] = {
                        f"{station}_EQ{i+1}": cycle_times[i] for i in range(num_eq)
                    }
                    group_names.append(station)

                    # Gather reliability information if columns exist
                    # Lowercase column names ensure matching
                    if {'issue count', 'downtime', 'duration'}.issubset(df.columns):
                        try:
                            issue_count = float(row['issue count']) if pd.notna(row['issue count']) else None
                        except Exception:
                            issue_count = None
                        try:
                            downtime_val = float(row['downtime']) if pd.notna(row['downtime']) else None
                        except Exception:
                            downtime_val = None
                        try:
                            duration_val = float(row['duration']) if pd.notna(row['duration']) else None
                        except Exception:
                            duration_val = None
                        if issue_count and issue_count > 0 and downtime_val is not None and duration_val is not None:
                            mtbf_g = max((duration_val - downtime_val) / issue_count, 0.0)
                            mttr_g = max(downtime_val / issue_count, 0.0)
                            group_metrics[station] = (mtbf_g, mttr_g)
                        else:
                            group_metrics[station] = None

                # Automatically set connections in sequence:
                connections = {}
                from_stations = {}

                for i, group in enumerate(group_names):
                    if i == 0:
                        from_stations[group] = []  # First station receives from START
                        connections[group] = [group_names[i + 1]] if i + 1 < len(group_names) else []
                    elif i == len(group_names) - 1:
                        from_stations[group] = [group_names[i - 1]]
                        connections[group] = []  # Last station sends to STOP
                    else:
                        from_stations[group] = [group_names[i - 1]]
                        connections[group] = [group_names[i + 1]]

                st.session_state.from_stations = from_stations
                st.session_state.connections = connections
                st.session_state.group_names = group_names

                upload_mode = True  # Indicate upload mode active

                st.success("Sheet processed and station connections auto-generated!")

            except Exception as e:
                st.error(f"Error processing file: {e}")
                return

    else:
        # Manual entry mode
        st.header("Step 1: Define Station Groups")
        num_groups = st.number_input("How many station groups?", min_value=1, step=1, key="num_groups_new")
        for i in range(num_groups):
            with st.expander(f"Station Group {i + 1}"):
                group_name = st.text_input(f"Group Name {i + 1}", key=f"group_name_{i}").strip().upper()
                if group_name:
                    num_eq = st.number_input(f"Number of Equipment in {group_name}", min_value=1, step=1, key=f"eq_count_{i}")
                    eq_dict = {}
                    for j in range(num_eq):
                        eq_name = f"{group_name}_EQ{j+1}"
                        cycle_time = st.number_input(f"Cycle Time for {eq_name} (sec)", min_value=0.1, key=f"ct_{i}_{j}")
                        eq_dict[eq_name] = cycle_time
                    valid_groups[group_name] = eq_dict
                    group_names.append(group_name)
                else:
                    group_names.append("")

        st.session_state.group_names = group_names

    # Step 2: Connections - ONLY show if manual entry (not upload mode)
    if not upload_mode:
        st.header("Step 2: Connect Stations")
        if "from_stations" not in st.session_state:
            st.session_state.from_stations = {}
        if "connections" not in st.session_state:
            st.session_state.connections = {}

        for i, name in enumerate(group_names):
            if not name:
                continue
            with st.expander(f"{name} Connections"):
                from_options = ['START'] + [g for g in group_names if g and g != name]
                to_options = ['STOP'] + [g for g in group_names if g and g != name]

                from_selected = st.multiselect(f"{name} receives from:", from_options, key=f"from_{i}")
                to_selected = st.multiselect(f"{name} sends to:", to_options, key=f"to_{i}")

                st.session_state.from_stations[name] = [] if "START" in from_selected else from_selected
                st.session_state.connections[name] = [] if "STOP" in to_selected else to_selected

    # Determine lockout zones based on group_names
    # filter out any empty group names from manual entry
    filtered_group_names = [g for g in group_names if g]
    lockout_zones = determine_lockout_zones(filtered_group_names)

    # Display lockout zones
    if lockout_zones:
        st.header("Lockout Zones Identification")
        for i, zone in enumerate(lockout_zones):
            st.markdown(f"**L{i+1}**: {', '.join(zone)}")

    # Ask whether to consider downtime and compute zone-level parameters accordingly
    consider_choice = st.radio("Consider downtime?", ["No", "Yes"], key="consider_downtime_choice")
    consider_downtime = (consider_choice == "Yes")
    zone_params = {}

    # If downtime is considered, gather reliability information and compute zone-level parameters
    if consider_downtime:
        # Build a local metrics dictionary for groups
        local_group_metrics = {}
        if method == "Upload Sheet":
            # Use reliability parsed from the uploaded sheet
            local_group_metrics = group_metrics.copy()
        else:
            # Manual entry: ask for issue count, downtime and duration per station group
            st.subheader("Provide Downtime Details per Station Group")
            for g in filtered_group_names:
                st.markdown(f"##### {g}")
                issues_val = st.number_input(
                    f"Number of Issues for {g}", min_value=0, step=1, key=f"issues_{g}")
                downtime_val = st.number_input(
                    f"Total Downtime for {g} (seconds)", min_value=0.0, step=1.0, key=f"downtime_{g}")
                duration_val = st.number_input(
                    f"Duration of Data for {g} (seconds)", min_value=1.0, step=1.0, key=f"duration_{g}")
                if issues_val and issues_val > 0:
                    mtbf_g = max((duration_val - downtime_val) / issues_val, 0.0)
                    mttr_g = max(downtime_val / issues_val, 0.0)
                    local_group_metrics[g] = (mtbf_g, mttr_g)
                else:
                    local_group_metrics[g] = None
        # Compute zone-level parameters from local_group_metrics
        for i, zone in enumerate(lockout_zones):
            zone_name = f"L{i+1}"
            total_lambda = 0.0
            avail_product = 1.0
            has_rel = False
            for g in zone:
                rel = local_group_metrics.get(g)
                eq_count = len(valid_groups.get(g, {}))
                if rel and rel[0] is not None:
                    mtbf_g, mttr_g = rel
                    if mtbf_g > 0:
                        for _ in range(eq_count):
                            total_lambda += 1.0 / mtbf_g
                            avail = mtbf_g / (mtbf_g + mttr_g) if (mtbf_g + mttr_g) > 0 else 1.0
                            avail_product *= avail
                    has_rel = True
            if has_rel and total_lambda > 0:
                zone_mtbf = 1.0 / total_lambda
                zone_avail = avail_product
                zone_mttr = zone_mtbf * (1.0 / zone_avail - 1.0) if zone_avail > 0 else None
                zone_params[zone_name] = {"mtbf": zone_mtbf, "mttr": zone_mttr}
            else:
                zone_params[zone_name] = None
    else:
        # No downtime considered: all zones have no failures
        for i in range(len(lockout_zones)):
            zone_params[f"L{i+1}"] = None

    # Step 3: Simulation Duration & Save (shown always)
    st.header("Step 3: Simulation Duration & Save")
    duration = st.number_input("Simulation Duration (seconds)", min_value=10, value=100, step=10, key="sim_duration_new")
    sim_name = st.text_input("Simulation Name", value="simulation_summary", key="sim_name_new").strip()
    if not sim_name:
        sim_name = "simulation_summary"

    st.header("Save your simulation setup")
    save_as = st.text_input("Filename to save current inputs", value=sim_name, key="save_filename")
    if st.button("💾 Save Current Setup"):
        # Persist simulation configuration along with computed zone parameters and downtime choice
        data_to_save = {
            "station_groups": [{"group_name": g, "equipment": valid_groups[g]} for g in valid_groups],
            "connections": [(src, dst) for src, tos in st.session_state.connections.items() for dst in tos],
            "from_stations": st.session_state.from_stations,
            "duration": duration,
            "simulation_name": save_as,
            "valid_groups": valid_groups,
            "lockout_zones": lockout_zones,
            "zone_params": zone_params,
            "consider_downtime": consider_downtime,
        }
        with open(os.path.join(SAVE_DIR, f"{save_as}.json"), "w") as f:
            json.dump(data_to_save, f, indent=2)
        st.success(f"Saved simulation as {save_as}.json")

    if st.button("▶️ Run Simulation"):
        station_groups_data = [{"group_name": g, "equipment": valid_groups[g]} for g in valid_groups]
        run_result = run_simulation_backend(
            station_groups_data,
            [(src, dst) for src, tos in st.session_state.connections.items() for dst in tos],
            st.session_state.from_stations,
            duration,
            lockout_zones=lockout_zones,
            zone_params=zone_params
        )
        show_detailed_summary(run_result, valid_groups, st.session_state.from_stations, duration)

def open_simulation():
    st.title("🛠️ Production Line Simulation App (Discrete Event Simulation)")
    st.subheader("📂 Open Simulation")

    col1, col2 = st.columns(2)
    if col1.button("🔙 Back"):
        st.session_state.page = "main"
        return
    if col2.button("🏠 Home"):
        st.session_state.page = "main"
        return

    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".json")]
    if not files:
        st.warning("No simulations found.")
        return

    display_names = []
    file_map = {}
    for f in files:
        try:
            with open(os.path.join(SAVE_DIR, f), "r") as jf:
                data = json.load(jf)
                display_name = data.get("simulation_name", f[:-5])
                display_names.append(display_name)
                file_map[display_name] = f
        except Exception:
            display_names.append(f[:-5])
            file_map[f[:-5]] = f

    selected_name = st.selectbox("Choose simulation to open:", display_names)
    if st.button("Open Selected Simulation"):
        filename = file_map[selected_name]
        with open(os.path.join(SAVE_DIR, filename), "r") as f:
            data = json.load(f)
        st.session_state.simulation_data = data
        st.session_state.page = "edit"

def edit_simulation():
    st.title("🛠️ Production Line Simulation App (Discrete Event Simulation)")
    sim_name = st.session_state.simulation_data.get("simulation_name", "Unnamed")
    st.subheader(f"✏️ Edit & Rerun Simulation: {sim_name}")

    col1, col2 = st.columns(2)
    if col1.button("🔙 Back"):
        st.session_state.page = "open"
        return
    if col2.button("🏠 Home"):
        st.session_state.page = "main"
        return

    # Show editable JSON text area
    json_str = json.dumps(st.session_state.simulation_data, indent=2)
    edited_json_str = st.text_area("Edit Simulation JSON here:", value=json_str, height=400)

    parse_error = None
    valid_json = None

    if st.button("Validate JSON"):
        try:
            valid_json = json.loads(edited_json_str)
            st.success("JSON is valid!")
            st.session_state.simulation_data = valid_json
        except Exception as e:
            parse_error = str(e)
            st.error(f"Invalid JSON: {parse_error}")

    # Use duration from edited JSON if valid, else from session data, else default 100
    duration_val = 100
    try:
        duration_val = st.session_state.simulation_data.get("duration", 100)
    except Exception:
        pass

    duration_val = st.number_input(
        "Simulation Duration (seconds)",
        min_value=10,
        value=duration_val,
        step=10,
        key="edit_duration"
    )

    if st.button("▶️ Run Simulation Again"):
        try:
            sim_data = st.session_state.simulation_data
            if isinstance(sim_data, str):
                sim_data = json.loads(sim_data)

            # Update duration with current input value
            sim_data["duration"] = duration_val
            st.session_state.simulation_data = sim_data

            lockout_zones = sim_data.get("lockout_zones")
            zone_params = sim_data.get("zone_params")
            # Convert zone_params into expected format (zone_name -> (mtbf, mttr)) if stored as dict
            converted_zone_params = None
            if zone_params:
                converted_zone_params = {}
                for k, v in zone_params.items():
                    if v and isinstance(v, dict):
                        converted_zone_params[k] = (v.get("mtbf"), v.get("mttr"))
                    else:
                        converted_zone_params[k] = None

            run_result = run_simulation_backend(
                sim_data["station_groups"],
                sim_data["connections"],
                sim_data["from_stations"],
                duration_val,
                lockout_zones=lockout_zones,
                zone_params=converted_zone_params
            )
            valid_groups = {g["group_name"]: g["equipment"] for g in sim_data["station_groups"]}
            show_detailed_summary(run_result, valid_groups, sim_data["from_stations"], duration_val)

        except Exception as e:
            st.error(f"Error running simulation: {e}")

    if st.button("💾 Save Edited Simulation"):
        try:
            sim_data = st.session_state.simulation_data
            if isinstance(sim_data, str):
                sim_data = json.loads(sim_data)

            sim_data["duration"] = duration_val
            save_as = sim_data.get("simulation_name", "edited_simulation")
            with open(os.path.join(SAVE_DIR, f"{save_as}.json"), "w") as f:
                json.dump(sim_data, f, indent=2)
            st.success(f"Saved edited simulation as {save_as}.json")
        except Exception as e:
            st.error(f"Error saving simulation: {e}")

# ========== Simulation Backend ==========

def run_simulation_backend(station_groups_data, connections_list, from_stations_dict, duration,
                           lockout_zones=None, zone_params=None, downtime_info=None):
    """
    Prepare and execute the simulation. Optional parameters allow specifying lockout zones and
    either precomputed zone-level MTBF/MTTR values (zone_params) or legacy downtime information.
    If zone_params is provided it will be used directly. If not, downtime_info will be used
    for backward compatibility to compute zone parameters. If neither is provided, zones will
    operate without downtime.
    """
    env = simpy.Environment()
    station_groups = {g["group_name"]: g["equipment"] for g in station_groups_data}

    # Convert connection list (tuples) into dictionary
    connections = defaultdict(list)
    for src, dst in connections_list:
        connections[src].append(dst)

    # Determine group-to-zone mapping
    if lockout_zones:
        group_to_zone = {}
        for i, zone in enumerate(lockout_zones):
            zone_name = f"L{i+1}"
            for g in zone:
                group_to_zone[g] = zone_name
    else:
        # If no lockout zones provided, treat all groups as a single zone L1
        group_to_zone = {}
        for g in station_groups.keys():
            group_to_zone[g] = "L1"
        lockout_zones = [list(station_groups.keys())]

    # Prepare zone parameters: use provided zone_params if available, else compute from downtime_info
    computed_zone_params = {}
    if zone_params is not None:
        # Convert any dict or tuple values into a uniform tuple or None
        for i, zone in enumerate(lockout_zones):
            zone_name = f"L{i+1}"
            param = zone_params.get(zone_name) if isinstance(zone_params, dict) else None
            if param is None:
                computed_zone_params[zone_name] = None
            elif isinstance(param, tuple) and len(param) == 2:
                computed_zone_params[zone_name] = param
            elif isinstance(param, dict):
                computed_zone_params[zone_name] = (param.get("mtbf"), param.get("mttr"))
            else:
                computed_zone_params[zone_name] = None
    elif downtime_info:
        # Legacy behaviour: compute zone parameters from downtime info
        for i, zone in enumerate(lockout_zones):
            zone_name = f"L{i+1}"
            info = downtime_info.get(zone_name) if isinstance(downtime_info, dict) else None
            if info and info.get("issues", 0) > 0:
                total_downtime = info.get("downtime", 0.0)
                num_issues = info.get("issues", 0)
                total_uptime = max(duration - total_downtime, 0.0001)
                mtbf = total_uptime / num_issues
                mttr = total_downtime / num_issues
                computed_zone_params[zone_name] = (mtbf, mttr)
            else:
                computed_zone_params[zone_name] = None
    else:
        # No downtime consideration: all zones have no failures
        for i, zone in enumerate(lockout_zones):
            zone_name = f"L{i+1}"
            computed_zone_params[zone_name] = None

    sim = FactorySimulation(env, station_groups, duration, dict(connections), from_stations_dict,
                            group_to_zone=group_to_zone, zone_params=computed_zone_params)
    env.process(sim.run())
    env.run(until=duration)
    return sim

# ========== Simulation Class ==========

class FactorySimulation:
    def __init__(self, env, station_groups, duration, connections, from_stations,
                 group_to_zone=None, zone_params=None):
        self.env = env
        self.station_groups = station_groups
        self.connections = connections
        self.from_stations = from_stations
        self.duration = duration

        self.buffers = defaultdict(lambda: simpy.Store(env))
        self.resources = {
            eq: simpy.Resource(env, capacity=1)
            for group in station_groups.values()
            for eq in group
        }
        self.cycle_times = {
            eq: ct
            for group in station_groups.values()
            for eq, ct in group.items()
        }
        self.equipment_to_group = {
            eq: group
            for group, eqs in station_groups.items()
            for eq in eqs
        }

        # Throughput and WIP tracking
        self.throughput_in = defaultdict(int)
        self.throughput_out = defaultdict(int)
        self.wip_over_time = defaultdict(list)
        self.time_points = []
        self.equipment_busy_time = defaultdict(float)
        self.board_id = 1
        self.wip_interval = 5

        # MTBF/MTTR handling
        self.group_to_zone = group_to_zone or {g: "L1" for g in station_groups}
        self.zone_params = zone_params or {}
        # Initialize zone down flags
        self.zone_down_map = {}
        for z in set(self.group_to_zone.values()):
            self.zone_down_map[z] = False
        # Start zone failure processes if parameters provided
        for zone, params in self.zone_params.items():
            if params:
                mtbf, mttr = params
                self.env.process(self.zone_failure_process(zone, mtbf, mttr))

        env.process(self.track_wip())

    def zone_failure_process(self, zone, mtbf, mttr):
        """
        Process that generates failures for a given zone based on exponential MTBF and erlang MTTR.
        When a failure occurs, the zone is marked down for the repair duration.
        """
        while True:
            # Time to next failure (exponential distribution)
            try:
                ttf = random.expovariate(1 / mtbf)
            except ZeroDivisionError:
                ttf = float('inf')
            yield self.env.timeout(ttf)
            # Zone goes down
            self.zone_down_map[zone] = True
            # Repair time using Erlang distribution (shape 2, scale mttr/2)
            repair_time = 0.0
            if mttr > 0:
                # Gamma distribution with shape=2, scale=mttr/2 yields mean = mttr
                repair_time = random.gammavariate(2, mttr / 2.0)
            yield self.env.timeout(repair_time)
            # Zone goes back up
            self.zone_down_map[zone] = False

    def equipment_worker(self, eq):
        group = self.equipment_to_group[eq]
        while True:
            zone_name = self.group_to_zone.get(group, "L1")
            # If zone is down, wait until it comes back up
            while self.zone_params and zone_name and self.zone_down_map.get(zone_name, False):
                yield self.env.timeout(1)
            # Wait for a board to arrive
            board = yield self.buffers[group].get()
            self.throughput_in[eq] += 1
            # Wait again if zone goes down before processing
            while self.zone_params and zone_name and self.zone_down_map.get(zone_name, False):
                yield self.env.timeout(1)
            # Acquire equipment resource
            with self.resources[eq].request() as req:
                yield req
                # If the zone goes down while waiting, pause
                while self.zone_params and zone_name and self.zone_down_map.get(zone_name, False):
                    yield self.env.timeout(1)
                # Determine processing time
                ct = 0.0
                group_upper = group.upper()
                is_stacker_or_wip = ("STACKER" in group_upper) or ("WIP CART" in group_upper) or ("WIPCART" in group_upper)
                if is_stacker_or_wip:
                    # Only apply cycle time if there is backlog after removing current board
                    queue_len = len(self.buffers[group].items)
                    if queue_len > 0:
                        ct = self.cycle_times[eq]
                else:
                    ct = self.cycle_times[eq]
                start = self.env.now
                if ct > 0:
                    yield self.env.timeout(ct)
                end = self.env.now
                self.equipment_busy_time[eq] += (end - start)
            # Board completed at this equipment
            self.throughput_out[eq] += 1
            # Pass board to connected groups
            for tgt in self.connections.get(group, []):
                yield self.buffers[tgt].put(board)

    def track_wip(self):
        while True:
            snapshot = {}
            for group in self.station_groups:
                in_count = sum(self.throughput_in[eq] for eq in self.station_groups[group])
                out_count = sum(self.throughput_out[eq] for eq in self.station_groups[group])
                snapshot[group] = max(0, in_count - out_count)

            current_time = self.env.now
            self.time_points.append(current_time)
            for group, wip in snapshot.items():
                self.wip_over_time[group].append(wip)

            # Stop tracking if simulation is over
            if current_time >= self.duration:
                break

            next_time = min(current_time + self.wip_interval, self.duration)
            yield self.env.timeout(next_time - current_time)

    def feeder(self):
        while True:
            for group, sources in self.from_stations.items():
                if not sources:  # Root stations only
                    board = f"B{self.board_id}"
                    self.board_id += 1
                    yield self.buffers[group].put(board)
            yield self.env.timeout(1)

    def run(self):
        for group in self.station_groups:
            for eq in self.station_groups[group]:
                self.env.process(self.equipment_worker(eq))
        self.env.process(self.feeder())
        yield self.env.timeout(0)

# ========== Summary Display ==========

def show_detailed_summary(sim, valid_groups, from_stations, duration):
    st.markdown("---")
    st.subheader("📊 Simulation Results Summary")

    groups = list(valid_groups.keys())
    agg = defaultdict(lambda: {'in': 0, 'out': 0, 'busy': 0, 'count': 0, 'cycle_times': [], 'wip': 0})

    for group in groups:
        eqs = valid_groups[group]
        for eq in eqs:
            agg[group]['in'] += sim.throughput_in.get(eq, 0)
            agg[group]['out'] += sim.throughput_out.get(eq, 0)
            agg[group]['busy'] += sim.equipment_busy_time.get(eq, 0)
            agg[group]['cycle_times'].append(sim.cycle_times.get(eq, 0))
            agg[group]['count'] += 1

        prev_out = sum(
            sim.throughput_out.get(eq, 0)
            for g in from_stations.get(group, [])
            for eq in valid_groups.get(g, [])
        )
        curr_in = agg[group]['in']
        agg[group]['wip'] = max(0, prev_out - curr_in)

    df = pd.DataFrame([{
        "Station Group": g,
        "Boards In": agg[g]['in'],
        "Boards Out": agg[g]['out'],
        "WIP": agg[g]['wip'],
        "Number of Equipment": agg[g]['count'],
        "Cycle Times (s)": ", ".join([f"{ct:.1f}" for ct in agg[g]['cycle_times']]),
        "Utilization (%)": round(100 * agg[g]['busy'] / (agg[g]['count'] * duration), 1) if agg[g]['count'] else 0.0
    } for g in groups])

    st.dataframe(df)

    # Downloadable Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(sim.wip_over_time).to_excel(writer, sheet_name="WIP_Over_Time", index=False)
    output.seek(0)
    st.download_button("📥 Download Summary as Excel", data=output, file_name="simulation_results.xlsx")

    # === Bottleneck Detection and Suggestion ===
    st.subheader("💡 Bottleneck Analysis and Suggestion")
    if 'agg' in locals() and 'valid_groups' in locals():
        min_out = float('inf')
        bottleneck_group = None
        for group in groups:
            out = agg[group]['out']
            if out < min_out:
                min_out = out
                bottleneck_group = group

        if bottleneck_group:
            eqs = valid_groups[bottleneck_group]
            avg_ct = sum(sim.cycle_times[eq] for eq in eqs) / len(eqs)
            base_out = agg[groups[-1]]['out']
            eq_count = len(eqs)
            new_out_bottleneck = (agg[bottleneck_group]['out'] / eq_count) * (eq_count + 1)
            estimated_final_out = base_out + (new_out_bottleneck - agg[bottleneck_group]['out']) * 0.7

            delta_b = round(new_out_bottleneck - agg[bottleneck_group]['out'])
            delta_final = round(estimated_final_out - base_out)

            st.markdown(
                f"If you **add 1 more equipment** to **{bottleneck_group}** with cycle time = **{round(avg_ct,1)} sec**, "
                f"you may increase its output by approximately **{delta_b} boards**, "
                f"and final output by approximately **{delta_final} boards** over {duration} seconds."
            )
    else:
        st.info("ℹ️ Run the simulation to get bottleneck suggestions.")

    # === Throughput & WIP Bar Chart ===
    st.subheader("📈 Throughput & WIP")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(groups))
    bw = 0.25
    in_vals = [agg[g]['in'] for g in groups]
    out_vals = [agg[g]['out'] for g in groups]
    wip_vals = [agg[g]['wip'] for g in groups]

    bars1 = ax.bar(x, in_vals, width=bw, label='In', color='skyblue')
    bars2 = ax.bar([i + bw for i in x], out_vals, width=bw, label='Out', color='lightgreen')
    bars3 = ax.bar([i + 2 * bw for i in x], wip_vals, width=bw, label='WIP', color='salmon')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{int(height)}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks([i + bw for i in x])
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    st.download_button("📥 Download Chart (PNG)", data=buf, file_name="throughput_wip.png", mime="image/png")

    # === MTBF & MTTR Distribution Charts ===
    st.subheader("📉 MTBF & MTTR Distributions by Zone")
    # Plot probability density functions for each zone if parameters are available
    try:
        zone_params = getattr(sim, 'zone_params', {})
    except Exception:
        zone_params = {}
    if zone_params:
        for zone_name, params in zone_params.items():
            if params and isinstance(params, tuple) and params[0] is not None and params[1] is not None:
                mtbf, mttr = params
                # Limit to positive values
                if mtbf and mttr and mtbf > 0 and mttr > 0:
                    # Generate x-values for MTBF (exponential) and MTTR (Erlang k=2)
                    x_mtbf = np.linspace(0, 5 * mtbf, 100)
                    y_mtbf = (1.0 / mtbf) * np.exp(-x_mtbf / mtbf)
                    x_mttr = np.linspace(0, 5 * mttr, 100)
                    scale = mttr / 2.0
                    # Erlang(k=2) PDF: f(x) = (x/(scale^2)) * exp(-x/scale)
                    y_mttr = (x_mttr / (scale ** 2)) * np.exp(-x_mttr / scale)
                    fig_dist, ax_dist = plt.subplots(figsize=(6, 3))
                    ax_dist.plot(x_mtbf, y_mtbf, label=f"{zone_name} MTBF (Exp)")
                    ax_dist.plot(x_mttr, y_mttr, label=f"{zone_name} MTTR (Erlang)")
                    ax_dist.set_title(f"Zone {zone_name} MTBF & MTTR Distributions")
                    ax_dist.set_xlabel("Time (s)")
                    ax_dist.set_ylabel("Density")
                    ax_dist.legend()
                    ax_dist.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig_dist)
            else:
                st.markdown(f"**{zone_name}:** No downtime considered or insufficient data to compute distributions.")
    else:
        st.info("No zone-level MTBF/MTTR information available.")

    # === ZIP Download of All Charts and Tables ===
    st.markdown("### 📦 Export All Results")

    zip_name = st.text_input("Enter ZIP file name", value="simulation_results")
    if st.button("📥 Download All as ZIP"):
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w") as zf:
            # Excel summary
            if output.getbuffer().nbytes > 0:
                zf.writestr("summary.xlsx", output.getvalue())

            # WIP Over Time Chart (optional, regenerate here if needed)
            wip_buf = BytesIO()
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            wip_df = pd.DataFrame(sim.wip_over_time)
            wip_df.plot(ax=ax2)
            ax2.set_title("WIP Over Time")
            fig2.tight_layout()
            fig2.savefig(wip_buf, format="png")
            wip_buf.seek(0)
            zf.writestr("WIP_Over_Time.png", wip_buf.getvalue())

            # Throughput & WIP Chart
            buf.seek(0)
            zf.writestr("Throughput_WIP_Bar.png", buf.getvalue())

            # Layout diagram (regenerate safely)
            try:
                if groups:
                    dot = Digraph()
                    dot.attr(rankdir="LR", size="8")
                    for group in groups:
                        dot.node(group, shape="box", style="filled", fillcolor="lightblue")
                    for i in range(len(groups) - 1):
                        dot.edge(groups[i], groups[i + 1])
                    layout_buf = BytesIO()
                    layout_buf.write(dot.pipe(format="png"))
                    layout_buf.seek(0)
                    zf.writestr("Production_Layout.png", layout_buf.getvalue())
            except Exception as e:
                st.warning(f"Could not include layout diagram in ZIP: {e}")

        mem_zip.seek(0)
        st.download_button(
            label="📦 Download All as ZIP",
            data=mem_zip,
            file_name=f"{zip_name.strip() or 'simulation_results'}.zip",
            mime="application/zip"
        )

# ========== Page Navigation ==========
if st.session_state.page == "login":
    login_page()
elif not st.session_state.authenticated:
    login_page()
elif st.session_state.page == "main":
    main_page()
elif st.session_state.page == "new":
    new_simulation()
elif st.session_state.page == "open":
    open_simulation()
elif st.session_state.page == "edit":
    edit_simulation()
