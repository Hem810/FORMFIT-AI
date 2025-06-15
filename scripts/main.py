import tkinter as tk                               #To create the frontent
from tkinter import ttk, messagebox, scrolledtext  #To create the frontent
import time
import cv2                                         #To take the video input and manipulate it
import numpy as np                                 #Manipulate the input image sizes
import math
import onnxruntime as ort                          #To run the HRNET_POSE Model locally on NPU
import os
import torch
from torchvision.transforms import v2              #To ammend the input and output values of model
from torchvision.transforms.functional import InterpolationMode #To ammend the input and output values of model
from pathlib import Path
import json
from PIL import Image, ImageTk

                                                  #Some Constants for model inputs
expected_shape=[1, 3, 256, 192]
input_image_height, input_image_width = expected_shape[2], expected_shape[3]
heatmap_height, heatmap_width = 64, 48
scaler_height = input_image_height/heatmap_height
scaler_width = input_image_width/heatmap_width
OPENCV_AVAILABLE = True

class PoseDetectorQNN:                             # Model definition class
    def __init__(self,
                 model_path=r"models\hrnet_pose.onnx",
                 qnn_backend_path=r"C:/Users/DFS/Desktop/gitrepo/nuget_packages/Microsoft.ML.OnnxRuntime.QNN.1.20.1/runtimes/win-x64/native/QnnHtp.dll",
                 input_size=(256, 192),
                 conf_threshold=0.3):
                                                   # Initialize configuration
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.lmList = []
        self.draw=1
        self.session = None

        self._initialize_model(model_path, qnn_backend_path) #Model Initialization
        self.keypoint_indices = {
            0: "nose",
            1: "left_eye", 2: "right_eye",
            3: "left_ear", 4: "right_ear",
            5: "left_shoulder", 6: "right_shoulder",
            7: "left_elbow", 8: "right_elbow",
            9: "left_wrist", 10: "right_wrist",
            11: "left_hip", 12: "right_hip",
            13: "left_knee", 14: "right_knee",
            15: "left_ankle", 16: "right_ankle"
        }                                                  #Model Output to body-joint dictionary

    
    def _initialize_model(self, model_path, backend_path):  #To initialize the Model
        providers = [                                       #Providers:first choice is NPU,CPU as a fallback if NPU path fails
            (
                "QNNExecutionProvider",
                {
                    "backend_type": "htp",
                }
            ),
            "CPUExecutionProvider"
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(           #Session declaration
                model_path,
                sess_options,
                providers=providers
            )

            self.inputs = self.session.get_inputs()        #Sample inputs
            self.outputs = self.session.get_outputs()      # sample outputs
            self.input_0 = self.inputs[0]  
            self.output_0 = self.outputs[0]

            self.transformer = v2.Compose([    
                v2.Resize(size=(expected_shape[2],expected_shape[3]),interpolation=InterpolationMode.BICUBIC),
                v2.ToDtype(torch.float32),
                v2.ToTensor()
            ])

        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")


    def _preprocess(self, img):                         #Preprocess the image for the model
        hwc_frame_pil = Image.fromarray(img)
        transform_frame = self.transformer(hwc_frame_pil)
        hwc_frame_np = transform_frame.permute(1,2,0).numpy()
        inference_frame = np.expand_dims(transform_frame, axis=0)
        return inference_frame

    def keypoint_processor(self,post_inference_array: np.ndarray, scaler_height: int, scaler_width: int) -> list:
        """Finds the keypoints in the given array and returns their coordinates"""
        keypoint_coordinates = []
        for keypoint in range(post_inference_array.shape[0]):
            heatmap = post_inference_array[keypoint]
                                                        # Find the index of the maximum value in the heatmap
            max_val_index = torch.argmax(heatmap)
                                                        # Convert the flat index to a 2D coordinates (row, column)
            img_height, img_width = torch.unravel_index(max_val_index, heatmap.shape)
                                                        # Scale the coordinates
            coords = (int(img_height*scaler_height), int(img_width*scaler_width))
            keypoint_coordinates.append(coords)
        return (keypoint_coordinates)

    def findPose(self, img, draw=True):
        """Uses the HRNET model on NPU for pose estimation"""
        inference_frame = self._preprocess(img)
        outputs = self.session.run(None, {self.input_0.name:inference_frame})
        output_tensor = torch.tensor(outputs).squeeze(0).squeeze(0)
        self.lmlist=self.keypoint_processor(output_tensor, scaler_height, scaler_width)

        if self.draw:
            frame=self._draw_landmarks(img)
            return frame

    def _draw_landmarks(self, img):                     #Marks the points found by the model
        hwc_frame_pil = Image.fromarray(img)
        transform_frame = self.transformer(hwc_frame_pil)
        hwc_frame_np = transform_frame.permute(1,2,0).numpy()
        frame = (hwc_frame_np*255).astype(np.uint8)
        frame = frame.copy()

        for (y,x) in self.lmlist:
            cv2.circle(frame, (x,y), radius=0, color=(0,0,255), thickness=-1)

        return frame

    def findPosition(self, img, draw=True):
        frame=self._draw_landmarks(img)
        return frame,self.lmlist

    def findAngle(self, img, p1, p2, p3, draw=True): 
        """Finds the angles between the three needed points and returns the acute angle"""
        try:
            y1, x1 = self.lmlist[p1]
            y2, x2 = self.lmlist[p2]
            y3, x3 = self.lmlist[p3]
        except:
            return None

        if -1 in (x1, y1, x2, y2, x3, y3):
            return None

        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) -
            math.atan2(y1 - y2, x1 - x2)
        )

        angle = angle + 360 if angle < 0 else angle

        if(angle>=200):
            angle=360-angle

        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1) #Draws the line connecting points
        cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 1)
        for (x, y) in [(x1, y1), (x2, y2), (x3, y3)]:
            cv2.circle(img, (x, y), 2, (0, 0, 255), cv2.FILLED)
        return angle





class WorkoutApp:                    # main WorkoutApp class
    def __init__(self, root):
        self.root = root
        self.root.title("FORMFIT AI")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2B2B2B')

                                    # Modern styling
        self.style = ttk.Style()
        self.style.theme_use('clam')

                                    # Configure modern color scheme
        self.colors = {
            'bg_primary': '#2B2B2B',
            'bg_secondary': '#3C3C3C',
            'bg_accent': '#4A4A4A',
            'text_primary': '#FFFFFF',
            'text_secondary': '#CCCCCC',
            'accent': '#00D4AA',
            'button_hover': '#00A085',
            'danger': '#FF6B6B',
            'warning': '#FFD93D',
            'success': '#6BCF7F'
        }

                                        # Configure ttk styles
        self.configure_styles()

        self.load_workout_data()       #Loads workout data from the json file

        self.create_main_container()   # Creates main container

        self.create_navigation()       # Create navigation

        self.create_pages()            # Creates the different pages of the UI

        self.show_page("home")         # Initialize with homepage
                 
        self.video_cap = None          # Video variables for opencv
        self.video_running = False
        self.current_frame = None

    def configure_styles(self):
        """Configure modern ttk styles"""
        # Configure button styles
        self.style.configure('Modern.TButton',
                           background=self.colors['accent'],
                           foreground=self.colors['text_primary'],
                           font=('Segoe UI', 10, 'bold'),
                           borderwidth=0,
                           relief='flat',
                           padding=(15, 10))

        self.style.map('Modern.TButton',
                      background=[('active', self.colors['button_hover']),
                                ('pressed', self.colors['button_hover'])])

        # Navigation button style
        self.style.configure('Nav.TButton',
                           background=self.colors['bg_secondary'],
                           foreground=self.colors['text_secondary'],
                           font=('Segoe UI', 11, 'bold'),
                           borderwidth=0,
                           relief='flat',
                           padding=(20, 15))

        self.style.map('Nav.TButton',
                      background=[('active', self.colors['accent']),
                                ('pressed', self.colors['accent'])],
                      foreground=[('active', self.colors['text_primary']),
                                ('pressed', self.colors['text_primary'])])

        # Frame styles
        self.style.configure('Modern.TFrame',
                           background=self.colors['bg_secondary'],
                           relief='flat',
                           borderwidth=1)

        # Label styles
        self.style.configure('Title.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['text_primary'],
                           font=('Segoe UI', 24, 'bold'))

        self.style.configure('Subtitle.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['text_secondary'],
                           font=('Segoe UI', 14))

        self.style.configure('Modern.TLabel',
                           background=self.colors['bg_secondary'],
                           foreground=self.colors['text_primary'],
                           font=('Segoe UI', 11))

    def load_workout_data(self):
        """Loads workout data from JSON file and convert to expected format"""
        try:
            with open(r'scripts\workouts.json', 'r') as f:
                raw_data = json.load(f)

            # The exercises array can be used as-is since it matches the expected format
            self.workout_data = {
                "exercises": raw_data.get("exercises", []),
                "workouts": []
            }

            # Convert the hierarchical workout structure to flat structure
            self.workout_data["workouts"] = self.convert_workouts_to_flat_structure(raw_data.get("workouts", {}))

        except FileNotFoundError:
            messagebox.showerror("Error", "workouts.json file not found!")
            self.workout_data = {"workouts": [], "exercises": []}
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load workout data: {str(e)}")
            self.workout_data = {"workouts": [], "exercises": []}

    def convert_workouts_to_flat_structure(self, hierarchical_workouts):
        """Convert hierarchical workout structure to flat array for compatibility"""
        flat_workouts = []

        for level, level_data in hierarchical_workouts.items():
            for body_part, body_part_data in level_data.items():
                if 'exercises' in body_part_data:
                    # Create a workout name based on level and body part
                    if level == "custom":
                        workout_name = f"Custom {body_part.replace('_', ' ').title()}"
                    else:
                        workout_name = f"{level.title()} {body_part.replace('_', ' ').title()}"

                    # Convert exercises to the expected format
                    exercises = []
                    for exercise in body_part_data['exercises']:
                        exercise_data = {
                            "name": exercise.get("name", "Unknown Exercise"),
                            "sets": exercise.get("sets", 3),
                            "reps": exercise.get("target_reps", exercise.get("target_time", 0)),
                            "type": exercise.get("type", "reps"),
                            "muscle_group": self.get_muscle_group_from_body_part(body_part),
                            "equipment": "None",  # Default since not specified in new format
                            # Keep original data for pose estimation
                            "original_data": exercise
                        }
                        exercises.append(exercise_data)

                    flat_workout = {
                        "name": workout_name,
                        "exercises": exercises,
                        "level": level,
                        "body_part": body_part
                    }
                    flat_workouts.append(flat_workout)
        return flat_workouts

    def get_muscle_group_from_body_part(self, body_part):
        """Map body part to muscle group"""
        mapping = {
            "upper_body": "Upper Body",
            "lower_body": "Lower Body", 
            "core": "Core"
        }
        return mapping.get(body_part, "Full Body")

    def save_workout_data(self):
        """Save workout data to JSON file keeping the original format"""
        try:
            # Convert flat structure back to hierarchical for saving
            hierarchical_data = {
                "workouts": {},
                "exercises": self.workout_data.get("exercises", [])
            }

            # Group workouts back by level and body part
            for workout in self.workout_data.get("workouts", []):
                level = workout.get("level", "custom")
                body_part = workout.get("body_part", workout['name'])

                if level not in hierarchical_data["workouts"]:
                    hierarchical_data["workouts"][level] = {}

                if body_part not in hierarchical_data["workouts"][level]:
                    hierarchical_data["workouts"][level][body_part] = {"exercises": []}

                # Convert exercises back to original format for saving
                for exercise in workout["exercises"]:
                    if "original_data" in exercise:
                        # Use the original_data which has been updated with custom sets/reps
                        hierarchical_data["workouts"][level][body_part]["exercises"].append(exercise["original_data"])
                    else:
                        # Fallback: create proper exercise structure
                        exercise_data = {
                            "name": exercise.get("name", "Unknown Exercise"),
                            "type": exercise.get("type", "reps"),
                            "sets": exercise.get("sets", 3),
                            "form_check": exercise.get("form_check", {}),
                            "rep_tracking": exercise.get("rep_tracking", {}),
                            "time_tracking": exercise.get("time_tracking", {})
                        }

                        # Add target_reps or target_time based on type
                        if exercise.get("type") == "reps":
                            exercise_data["target_reps"] = exercise.get("reps", exercise.get("target_reps", 10))
                        elif exercise.get("type") == "time":
                            exercise_data["target_time"] = exercise.get("target_time", 30)

                        hierarchical_data["workouts"][level][body_part]["exercises"].append(exercise_data)

            # Save to workouts.json
            with open(r'scripts\workouts.json', 'w') as f:
                json.dump(hierarchical_data, f, indent=4)
            self.load_workout_data()
            self.create_selector_page()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save workout data: {str(e)}")

    def create_main_container(self):
        """Create main container frame"""
        self.main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        self.main_container.pack(fill='both', expand=True, padx=10, pady=10)

    def create_navigation(self):
        """Creates the navigation bar"""
        nav_frame = ttk.Frame(self.main_container, style='Modern.TFrame')
        nav_frame.pack(fill='x', pady=(0, 10))

        # Navigation buttons
        nav_buttons = [
            ("🏠 Home", "home"),
            ("💪 Workout Generator", "generator"),
            ("📋 Workout Selector", "selector"),
            ("▶️ Workout", "workout")
        ]

        for text, page in nav_buttons:
            btn = ttk.Button(nav_frame, text=text, style='Nav.TButton',
                           command=lambda p=page: self.show_page(p))
            btn.pack(side='left', padx=5, pady=10)

    def create_pages(self):
        """Creates all application pages"""
        # Pages container
        self.pages_container = tk.Frame(self.main_container, bg=self.colors['bg_primary'])
        self.pages_container.pack(fill='both', expand=True)

        self.pages = {}

        # Create individual pages
        self.create_home_page()
        self.create_generator_page()
        self.create_selector_page()
        self.create_workout_page()

    def create_home_page(self):
        """Create homepage with welcome message"""
        home_frame = tk.Frame(self.pages_container, bg=self.colors['bg_primary'])

        # Welcome message
        welcome_frame = ttk.Frame(home_frame, style='Modern.TFrame')
        welcome_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Title
        title_label = ttk.Label(welcome_frame, text="Welcome to FORMFIT AI",
                              style='Title.TLabel')
        title_label.pack(pady=(50, 20))

        # Subtitle
        subtitle_label = ttk.Label(welcome_frame,
                                 text="Your complete fitness companion for managing workouts and exercises",
                                 style='Subtitle.TLabel')
        subtitle_label.pack(pady=(0, 30))

        # Features list
        features_frame = tk.Frame(welcome_frame, bg=self.colors['bg_secondary'])
        features_frame.pack(pady=20, padx=50, fill='x')

        features = [
            "🏋️ Generate custom workouts with your preferred exercises",
            "📚 Browse pre-made workout routines from our database",
            "▶️ Interactive workout sessions with video support",
            "📊 Track your sets, reps, and progress"
        ]

        for feature in features:
            feature_label = tk.Label(features_frame, text=feature,
                                   bg=self.colors['bg_secondary'],
                                   fg=self.colors['text_primary'],
                                   font=('Segoe UI', 12),
                                   anchor='w')
            feature_label.pack(pady=10, padx=20, fill='x')

        # Quick start button
        start_btn = ttk.Button(welcome_frame, text="🚀 Start Your Workout Journey",
                             style='Modern.TButton',
                             command=lambda: self.show_page("generator"))
        start_btn.pack(pady=30)

        self.pages['home'] = home_frame

    def create_generator_page(self):
        """Create workout generator page"""
        generator_frame = tk.Frame(self.pages_container, bg=self.colors['bg_primary'])

        # Main content frame
        content_frame = ttk.Frame(generator_frame, style='Modern.TFrame')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_label = ttk.Label(content_frame, text="Workout Generator", style='Title.TLabel')
        title_label.pack(pady=(20, 30))

        # Create two columns
        columns_frame = tk.Frame(content_frame, bg=self.colors['bg_secondary'])
        columns_frame.pack(fill='both', expand=True, padx=20)

        # Left column - Exercise selection
        left_frame = tk.Frame(columns_frame, bg=self.colors['bg_secondary'])
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Exercise selection
        exercise_label = ttk.Label(left_frame, text="Select Exercises:", style='Modern.TLabel')
        exercise_label.pack(pady=(20, 10), anchor='w')

        # Exercise listbox with scrollbar
        exercise_frame = tk.Frame(left_frame, bg=self.colors['bg_secondary'])
        exercise_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.exercise_listbox = tk.Listbox(exercise_frame,
                                         bg=self.colors['bg_accent'],
                                         fg=self.colors['text_primary'],
                                         font=('Segoe UI', 10),
                                         selectmode='multiple',
                                         height=15)

        exercise_scrollbar = tk.Scrollbar(exercise_frame)
        exercise_scrollbar.pack(side='right', fill='y')
        self.exercise_listbox.pack(side='left', fill='both', expand=True)
        self.exercise_listbox.config(yscrollcommand=exercise_scrollbar.set)
        exercise_scrollbar.config(command=self.exercise_listbox.yview)

        # Populate exercise list
        for exercise in self.workout_data.get('exercises', []):
            self.exercise_listbox.insert('end', f"{exercise['name']} ({exercise['muscle_group']})\n")

        # Right column - Sets and reps and customization
        right_frame = tk.Frame(columns_frame, bg=self.colors['bg_secondary'])
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Default Sets and Reps inputs
        inputs_label = ttk.Label(right_frame, text="Default Workout Parameters:", style='Modern.TLabel')
        inputs_label.pack(pady=(20, 10), anchor='w')

        # Sets input
        sets_frame = tk.Frame(right_frame, bg=self.colors['bg_secondary'])
        sets_frame.pack(fill='x', pady=10)
        sets_label = ttk.Label(sets_frame, text="Default Sets:", style='Modern.TLabel')
        sets_label.pack(side='left')
        self.sets_var = tk.StringVar(value="3")
        sets_entry = tk.Entry(sets_frame, textvariable=self.sets_var,
                             bg=self.colors['bg_accent'],
                             fg=self.colors['text_primary'],
                             font=('Segoe UI', 11),
                             width=10)
        sets_entry.pack(side='right')

        # Reps input
        reps_frame = tk.Frame(right_frame, bg=self.colors['bg_secondary'])
        reps_frame.pack(fill='x', pady=10)
        reps_label = ttk.Label(reps_frame, text="Default Reps:", style='Modern.TLabel')
        reps_label.pack(side='left')
        self.reps_var = tk.StringVar(value="12")
        reps_entry = tk.Entry(reps_frame, textvariable=self.reps_var,
                             bg=self.colors['bg_accent'],
                             fg=self.colors['text_primary'],
                             font=('Segoe UI', 11),
                             width=10)
        reps_entry.pack(side='right')

        # Generated workout display
        workout_label = ttk.Label(right_frame, text="Generated Workout:", style='Modern.TLabel')
        workout_label.pack(pady=(30, 10), anchor='w')

        self.generated_workout_text = scrolledtext.ScrolledText(right_frame,
                                                              bg=self.colors['bg_accent'],
                                                              fg=self.colors['text_primary'],
                                                              font=('Segoe UI', 10),
                                                              height=10,
                                                              width=30)
        self.generated_workout_text.pack(fill='both', expand=True)

        # Generate button
        generate_btn = ttk.Button(right_frame, text="🔥 Generate Workout",
                                style='Modern.TButton',
                                command=self.generate_workout)
        generate_btn.pack(pady=20)

        self.pages['generator'] = generator_frame

    def customize_exercises(self):
        """Open a dialog to customize sets and reps for each selected exercise"""
        selected_indices = self.exercise_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one exercise!")
            return None

        try:
            default_sets = int(self.sets_var.get())
            default_reps = int(self.reps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for default sets and reps!")
            return None

        # Create a dialog for customizing sets and reps
        customize_dialog = tk.Toplevel(self.root)
        customize_dialog.title("Customize Exercises")
        customize_dialog.geometry("600x500")
        customize_dialog.configure(bg=self.colors['bg_primary'])
        customize_dialog.transient(self.root)
        customize_dialog.grab_set()

        # Add a frame for exercises
        exercise_frame = ttk.Frame(customize_dialog, style='Modern.TFrame')
        exercise_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Add a title
        title_label = ttk.Label(exercise_frame, text="Customize Sets and Reps",
                              style='Title.TLabel')
        title_label.pack(pady=(10, 20))

        # Create a canvas with scrollbar for exercises
        canvas_frame = tk.Frame(exercise_frame, bg=self.colors['bg_secondary'])
        canvas_frame.pack(fill='both', expand=True)

        # Add scrollbar
        scrollbar = tk.Scrollbar(canvas_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')

        # Create canvas
        canvas = tk.Canvas(canvas_frame, bg=self.colors['bg_secondary'],
                          yscrollcommand=scrollbar.set,
                          highlightthickness=0)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=canvas.yview)

        # Create frame inside canvas for exercises
        inner_frame = tk.Frame(canvas, bg=self.colors['bg_secondary'])
        canvas.create_window((0, 0), window=inner_frame, anchor='nw', width=canvas.winfo_width())

        # Create variables to store sets and reps for each exercise
        custom_sets_vars = []
        custom_reps_vars = []

        # Add exercise customization widgets
        for i, index in enumerate(selected_indices):
            exercise_info = self.workout_data['exercises'][index]

            # Exercise frame
            ex_frame = tk.Frame(inner_frame, bg=self.colors['bg_accent'], bd=1, relief='solid')
            ex_frame.pack(fill='x', padx=10, pady=5)

            # Exercise name and info
            name_label = ttk.Label(ex_frame,
                                 text=f"{i+1}. {exercise_info['name']} ({exercise_info['muscle_group']})",
                                 style='Modern.TLabel')
            name_label.grid(row=0, column=0, columnspan=4, sticky='w', padx=10, pady=5)

            # Sets
            sets_label = ttk.Label(ex_frame, text="Sets:", style='Modern.TLabel')
            sets_label.grid(row=1, column=0, sticky='w', padx=10, pady=5)
            sets_var = tk.StringVar(value=str(default_sets))
            custom_sets_vars.append(sets_var)
            sets_entry = tk.Entry(ex_frame, textvariable=sets_var, width=5,
                                bg=self.colors['bg_accent'],
                                fg=self.colors['text_primary'],
                                font=('Segoe UI', 10))
            sets_entry.grid(row=1, column=1, padx=5, pady=5)

            # Reps
            reps_label = ttk.Label(ex_frame, text="Reps:", style='Modern.TLabel')
            reps_label.grid(row=1, column=2, sticky='w', padx=10, pady=5)
            reps_var = tk.StringVar(value=str(default_reps))
            custom_reps_vars.append(reps_var)
            reps_entry = tk.Entry(ex_frame, textvariable=reps_var, width=5,
                                bg=self.colors['bg_accent'],
                                fg=self.colors['text_primary'],
                                font=('Segoe UI', 10))
            reps_entry.grid(row=1, column=3, padx=5, pady=5)

        # Update canvas scrollregion when inner_frame changes size
        inner_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        # Buttons frame
        buttons_frame = tk.Frame(exercise_frame, bg=self.colors['bg_secondary'])
        buttons_frame.pack(fill='x', pady=0)
        
        name_frame = tk.Frame(buttons_frame, bg=self.colors['bg_secondary'])
        name_frame.pack(fill='x', pady=10)
        name_label = ttk.Label(name_frame, text="Workout Name:", style='Modern.TLabel')
        name_label.pack(side='left', padx=5)
        workout_name_var = tk.StringVar(value="Custom Workout")
        name_entry = tk.Entry(name_frame, textvariable=workout_name_var, width=20,
                             bg=self.colors['bg_accent'],
                             fg=self.colors['text_primary'],
                             font=('Segoe UI', 10))
        name_entry.pack(side='left', padx=5)
        
        # Save to JSON checkbox
        save_var = tk.BooleanVar(value=True)
        save_check = tk.Checkbutton(buttons_frame, text="Save to workout.json",
                                   variable=save_var,
                                   bg=self.colors['bg_secondary'],
                                   fg=self.colors['text_primary'],
                                   selectcolor=self.colors['bg_accent'],
                                   activebackground=self.colors['bg_secondary'],
                                   activeforeground=self.colors['text_primary'])
        save_check.pack(pady=10)

        # Result container
        result = {"confirmed": False, "exercises": [], "save": False, "name": ""}

        # Buttons
        def on_confirm():
            # Validate inputs
            try:
                for i, (sets_var, reps_var) in enumerate(zip(custom_sets_vars, custom_reps_vars)):
                    sets = int(sets_var.get())
                    reps = int(reps_var.get())
                    if sets <= 0 or reps <= 0:
                        raise ValueError("Sets and reps must be positive numbers")

                    result["exercises"].append({
                        "index": selected_indices[i],
                        "sets": sets,
                        "reps": reps
                    })

                result["confirmed"] = True
                result["save"] = save_var.get()
                result["name"] = workout_name_var.get().strip() or "Custom Workout"
                customize_dialog.destroy()

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {str(e)}")

        def on_cancel(): #Cancel the generation of the workout
            customize_dialog.destroy()

        confirm_btn = ttk.Button(buttons_frame, text="Confirm", style='Modern.TButton', command=on_confirm)
        confirm_btn.pack(side='right', padx=10)

        cancel_btn = ttk.Button(buttons_frame, text="Cancel", style='Modern.TButton', command=on_cancel)
        cancel_btn.pack(side='right', padx=10)

        # Waits for dialog to close
        self.root.wait_window(customize_dialog)

        return result if result["confirmed"] else None

    def generate_workout(self):
        """Generate workout based on selected exercises with custom sets and reps"""
        # Open customization dialog
        result = self.customize_exercises()
        if not result:
            return

        # Clear previous workout
        self.generated_workout_text.delete(1.0, 'end')

        # Generate workout text
        workout_text = f"🏋️ {result['name']} 🏋️\n"
        workout_text += "=" * 40 + "\n\n"

        # Create workout object for saving
        workout_obj = {
            "name": result['name'],
            "exercises": [],
            "level": "custom",
            "body_part": result['name']
        }

        for i, exercise_data in enumerate(result["exercises"], 1):
            index = exercise_data["index"]
            sets = exercise_data["sets"]
            reps = exercise_data["reps"]
            exercise_info = self.workout_data['exercises'][index]

            workout_text += f"{i}. {exercise_info['name']}\n"
            workout_text += f"   💪 Muscle Group: {exercise_info['muscle_group']}\n"
            workout_text += f"   🔢 Sets: {sets} | Reps: {reps}\n"
            workout_text += f"   🏃 Equipment: {exercise_info['equipment']}\n\n"

            # Create exercise object that preserves original pose estimation data
            original_data_copy = exercise_info.copy()
            original_data_copy['sets'] = sets

            # Update target_reps or target_time based on exercise type
            if exercise_info.get('type') == 'reps':
                original_data_copy['target_reps'] = reps
            elif exercise_info.get('type') == 'time':
                original_data_copy['target_time'] = reps
            else:
                original_data_copy['target_reps'] = reps

            custom_exercise = {
                "name": exercise_info['name'],
                "sets": sets,
                "reps": reps,
                "type": exercise_info.get('type', 'reps'),
                "muscle_group": exercise_info['muscle_group'],
                "equipment": exercise_info['equipment'],
                # Keep original data for pose estimation with updated values
                "original_data": original_data_copy
            }

            workout_obj["exercises"].append(custom_exercise)

        workout_text += "=" * 40 + "\n"
        workout_text += "💡 Tip: Take 1-2 minutes rest between sets!"

        # Display the workout
        self.generated_workout_text.insert(1.0, workout_text)

        # Save workout if requested
        if result["save"]:
            self.workout_data["workouts"].append(workout_obj)
            self.save_workout_data()
            messagebox.showinfo("Success", f"Workout '{result['name']}' saved successfully!")

    def create_selector_page(self):
        """Create workout selector page"""
        selector_frame = tk.Frame(self.pages_container, bg=self.colors['bg_primary'])

        # Main content frame
        content_frame = ttk.Frame(selector_frame, style='Modern.TFrame')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_label = ttk.Label(content_frame, text="Workout Selector", style='Title.TLabel')
        title_label.pack(pady=(20, 30))

        # Create a canvas with scrollbar for workout list
        canvas_frame = tk.Frame(content_frame, bg=self.colors['bg_secondary'])
        canvas_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Add scrollbar
        scrollbar = tk.Scrollbar(canvas_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')

        # Create canvas
        canvas = tk.Canvas(canvas_frame, bg=self.colors['bg_secondary'],
                          yscrollcommand=scrollbar.set,
                          highlightthickness=0)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=canvas.yview)

        # Create frame inside canvas for workout cards
        list_frame = tk.Frame(canvas, bg=self.colors['bg_secondary'])
        canvas.create_window((0, 0), window=list_frame, anchor='nw')

        # Create workout cards
        for i, workout in enumerate(self.workout_data.get('workouts', [])):
            self.create_workout_card(list_frame, workout, i)

        # Update canvas scrollregion when list_frame changes size
        list_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        self.pages['selector'] = selector_frame

    def create_workout_card(self, parent, workout, index):
        """Create a workout card"""
        card_frame = tk.Frame(parent, bg=self.colors['bg_accent'], relief='raised', bd=1)
        card_frame.pack(fill='x', pady=10, padx=10)

        # Card header
        header_frame = tk.Frame(card_frame, bg=self.colors['bg_accent'])
        header_frame.pack(fill='x', padx=15, pady=(15, 5))

        workout_name = tk.Label(header_frame, text=workout['name'],
                               bg=self.colors['bg_accent'],
                               fg=self.colors['text_primary'],
                               font=('Segoe UI', 16, 'bold'))
        workout_name.pack(side='left')

        select_btn = ttk.Button(header_frame, text="Select Workout",
                              style='Modern.TButton',
                              command=lambda w=workout: self.select_workout(w))
        select_btn.pack(side='right')

        # Exercise list
        exercises_frame = tk.Frame(card_frame, bg=self.colors['bg_accent'])
        exercises_frame.pack(fill='x', padx=15, pady=(5, 15))

        exercises_text = "Exercises: "
        for j, exercise in enumerate(workout['exercises']):
            if j > 0:
                exercises_text += ", "
            exercises_text += f"{exercise['name']} ({exercise['sets']}x{exercise['reps']})"

        exercises_label = tk.Label(exercises_frame, text=exercises_text,
                                 bg=self.colors['bg_accent'],
                                 fg=self.colors['text_secondary'],
                                 font=('Segoe UI', 10),
                                 wraplength=800,
                                 justify='left')
        exercises_label.pack(anchor='w')

    def select_workout(self, workout):
        """Select a workout and navigate to workout page"""
        self.selected_workout = workout
        self.show_page('workout')
        self.update_workout_display()

    def create_workout_page(self):
        """Create workout execution page"""
        workout_frame = tk.Frame(self.pages_container, bg=self.colors['bg_primary'])

        # Main content frame
        content_frame = ttk.Frame(workout_frame, style='Modern.TFrame')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_label = ttk.Label(content_frame, text="Workout Session", style='Title.TLabel')
        title_label.pack(pady=(20, 20))

        # Create main layout - two columns
        main_layout = tk.Frame(content_frame, bg=self.colors['bg_secondary'])
        main_layout.pack(fill='both', expand=True, padx=20)

        # Left column - Video and controls
        left_column = tk.Frame(main_layout, bg=self.colors['bg_secondary'])
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Video frame
        video_label = ttk.Label(left_column, text="Video Frame:", style='Modern.TLabel')
        video_label.pack(pady=(20, 10), anchor='w')

        self.video_frame = tk.Frame(left_column, bg=self.colors['bg_accent'],
                                   width=400, height=300, relief='sunken', bd=2)
        self.video_frame.pack(pady=(0, 20))
        self.video_frame.pack_propagate(False)

        # Video placeholder
        self.video_label = tk.Label(self.video_frame,
                                   text="📹 Video Feed\n(OpenCV integration ready)",
                                   bg=self.colors['bg_accent'],
                                   fg=self.colors['text_secondary'],
                                   font=('Segoe UI', 14))
        self.video_label.pack(expand=True)

        # Control buttons
        controls_frame = tk.Frame(left_column, bg=self.colors['bg_secondary'])
        controls_frame.pack(fill='x', pady=10)

        self.start_btn = ttk.Button(controls_frame, text="▶️ Start",
                                  style='Modern.TButton',
                                  command=self.start_workout)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(controls_frame, text="⏹️ Stop",
                                 style='Modern.TButton',
                                 command=self.stop_workout)
        self.stop_btn.pack(side='left', padx=5)

        self.next_btn = ttk.Button(controls_frame, text="⏭️ Next",
                                 style='Modern.TButton',
                                 command=self.next_exercise)
        self.next_btn.pack(side='left', padx=5)

        # Right column - Workout info and text area
        right_column = tk.Frame(main_layout, bg=self.colors['bg_secondary'])
        right_column.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Current workout display
        workout_info_label = ttk.Label(right_column, text="Current Workout:", style='Modern.TLabel')
        workout_info_label.pack(pady=(20, 10), anchor='w')

        self.current_workout_text = scrolledtext.ScrolledText(right_column,
                                                            bg=self.colors['bg_accent'],
                                                            fg=self.colors['text_primary'],
                                                            font=('Segoe UI', 11),
                                                            height=10,
                                                            width=35)
        self.current_workout_text.pack(fill='x', pady=(0, 20))

        # Backend text area
        backend_label = ttk.Label(right_column, text="Workout Notes:", style='Modern.TLabel')
        backend_label.pack(pady=(10, 10), anchor='w')

        self.backend_text = scrolledtext.ScrolledText(right_column,
                                                    bg=self.colors['bg_accent'],
                                                    fg=self.colors['text_primary'],
                                                    font=('Segoe UI', 10),
                                                    height=8,
                                                    width=35)
        self.backend_text.pack(fill='both', expand=True)

        # Initialize with placeholder text
        self.backend_text.insert(1.0, "💡 Workout tips and instructions will appear here...\n\n")
        self.backend_text.insert('end', "🔥 Stay motivated and push your limits!\n")
        self.backend_text.insert('end', "💪 Remember to maintain proper form\n")
        self.backend_text.insert('end', "🧘 Focus on your breathing\n")
        self.backend_text.insert('end', "⏱️ Take adequate rest between sets\n")

        self.pages['workout'] = workout_frame
        self.selected_workout = None
        self.current_exercise_index = 0

    def update_workout_display(self):
        """Update the workout display with selected workout"""
        if hasattr(self, 'selected_workout') and self.selected_workout:
            self.current_workout_text.delete(1.0, 'end')
            workout_text = f"🏋️ {self.selected_workout['name']}\n"
            workout_text += "=" * 40 + "\n\n"

            for i, exercise in enumerate(self.selected_workout['exercises'], 1):
                status = "👆 CURRENT" if i == self.current_exercise_index + 1 else ""
                workout_text += f"{i}. {exercise['name']} {status}\n"
                workout_text += f"   Sets: {exercise['sets']} | Reps: {exercise['reps']}\n\n"

            self.current_workout_text.insert(1.0,workout_text)

    def start_workout(self):
        """Starts the workout session"""
        self.idx = 0                #Important variables initialization
        self.current_set = 1
        self.rep_count = 0
        self.direction = 0
        self.start_time = None
        self.pTime = 0
        self.counter=[True,0]
        self.counter2=[True,0]
        self.backend_text.insert('end', f"\n🚀 Workout started at {time.strftime('%H:%M:%S')}\n")
        self.backend_text.see('end')

        self.detector = PoseDetectorQNN(    #Model intialization call
            model_path=r"models\hrnet_pose.onnx",
            qnn_backend_path=r"C:/Users/DFS/Desktop/gitrepo/nuget_packages/Microsoft.ML.OnnxRuntime.QNN.1.20.1/runtimes/win-x64/native/QnnHtp.dll"
        )

        # Get exercises from selected workout for pose estimation
        if hasattr(self, 'selected_workout') and self.selected_workout:
            self.exercises = []
            for exercise in self.selected_workout['exercises']:
                # Check if exercise has original_data for pose estimation
                if 'original_data' in exercise:
                    self.exercises.append(exercise['original_data'])
                else:
                    # Create basic structure for pose estimation if not available
                    self.exercises.append({
                        "name": exercise['name'],
                        "type": "reps",
                        "target_reps": exercise.get('reps', 10),
                        "sets": exercise.get('sets', 3),
                        "form_check": {},
                        "rep_tracking": {
                            "count_angle": {
                                "points": [5, 7, 9],  # Default arm points
                                "up_position": [35, 55],
                                "down_position": [80, 100],
                                "threshold": 10
                            }
                        }
                    })

        if OPENCV_AVAILABLE:
            self.start_video_feed()
        else:
            self.video_label.config(text="📹 Workout Started!\n(Demo mode - OpenCV not available)")

    def stop_workout(self):  
        """Stops workout session"""
        self.backend_text.insert('end', f"\n⏹️ Workout stopped at {time.strftime('%H:%M:%S')}\n")
        self.backend_text.see('end')

        if OPENCV_AVAILABLE and self.video_cap:
            self.video_running = False
            self.video_cap.release()
            self.video_cap = None
            self.video_label.config(text="📹 Workout Stopped\n(Video feed ended)")

    def next_exercise(self):
        """Moves to next exercise"""
        if hasattr(self, 'selected_workout') and self.selected_workout:
            if self.current_exercise_index < len(self.selected_workout['exercises']) - 1:
                self.current_exercise_index += 1
                self.idx = self.current_exercise_index  # Update pose estimation index
                self.current_set = 1
                self.rep_count = 0
                self.direction = 0

                self.update_workout_display()
                current_ex = self.selected_workout['exercises'][self.current_exercise_index]
                self.backend_text.insert('end', f"\n⏭️ Next exercise: {current_ex['name']}\n")
                self.backend_text.see('end')
            else:
                self.backend_text.insert('end', "\n🎉 Workout completed! Great job!\n")
                self.backend_text.see('end')

    def start_video_feed(self):
        """Starts a video feed using OpenCV (if available)"""
        if OPENCV_AVAILABLE:
            try:
                # Try to use default camera (0)
                self.video_cap = cv2.VideoCapture(0)
                if self.video_cap.isOpened():
                    self.video_running = True
                    self.update_video_frame()
                else:
                    self.video_label.config(text="📹 Camera not found\n(Using demo mode)")
            except Exception as e:
                self.video_label.config(text=f"📹 Camera error\n{str(e)}")

    def update_video_frame(self):
        """Update the video frame"""
        if self.video_running and self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                # Resize frame to fit the display area
                frame_rgb=self.pose_esti_current(frame)
                image = Image.fromarray(frame_rgb)

                image = image.resize((380, 280), Image.LANCZOS) 

                photo = ImageTk.PhotoImage(image)
                
                # Update label
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo

                # Schedule next frame update
                self.root.after(30, self.update_video_frame)

    def show_page(self, page_name):
        """Show the specific page"""
        # Hide all pages
        for page in self.pages.values():
            page.pack_forget()

        # Show selected page
        if page_name in self.pages:
            self.pages[page_name].pack(fill='both', expand=True)

    def pose_esti_current(self, img):
        """Main function that finds the pose and angles and then uses logic to walkthrough the entire workout """

        img = self.detector.findPose(img)
        frame, lmList = self.detector.findPosition(img, draw=True)

        if hasattr(self, 'exercises') and self.idx < len(self.exercises):
            exercise = self.exercises[self.idx]
            name = exercise.get("name", "Unknown Exercise")
            form = exercise.get("form_check", {})

            # Primary angle check
            pa = form.get("primary_angle", {})
            if pa.get("points"):
                angle1 = self.detector.findAngle(img, *pa.get("points", []), draw=True)
                if angle1 is not None and (angle1 < pa.get("min_angle", 0) or angle1 > pa.get("max_angle", 360)):
                    if(self.counter[0]):
                        self.counter[1]=0
                        self.counter[0]=False
                    else:
                        self.counter[1]+=1
                    if(self.counter[1]==15):
                        self.backend_text.insert('end', f"Warning [{name}]: {pa.get('description', 'Primary form incorrect')}\n")
                        self.backend_text.see('end')
                if(angle1 is not None and (angle1 >= pa.get("min_angle", 0) and angle1 <= pa.get("max_angle", 360))):
                    if(not self.counter[0]):
                        self.counter[1]=0
                        self.counter[0]=True
                    else:
                        self.counter[1]+=1
                    if(self.counter[1]==15):
                        self.backend_text.insert('end', f"Position is now correct\n")
                        self.backend_text.see('end')

            # Secondary angle check
            sa = form.get("secondary_angle", {})
            if sa.get("points"):
                angle2 = self.detector.findAngle(img, *sa.get("points", []), draw=True)
                if(angle2 is not None and (angle2 < sa.get("min_angle", 0) or angle2 > sa.get("max_angle", 360))):
                    if(self.counter2[0]):
                        self.counter2[1]=0
                        self.counter2[0]=False
                    else:
                        self.counter2[1]+=1
                    if(self.counter2[1]==15):
                        self.backend_text.insert('end', f"Warning [{name}]: {sa.get('description', 'Secondary form incorrect')}\n")
                        self.backend_text.see('end')
                if(angle2 is not None and (angle2 >= sa.get("min_angle", 0) and angle2 <= sa.get("max_angle", 360))):
                    if(not self.counter2[0]):
                        self.counter2[1]=0
                        self.counter2[0]=True
                    else:
                        self.counter2[1]+=1
                    if(self.counter2[1]==15):
                        self.backend_text.insert('end', f"Position is now correct\n")
                        self.backend_text.see('end')
            # Exercise type handling
            etype = exercise.get("type")
            if etype == "reps":
                rt = exercise.get("rep_tracking", {}).get("count_angle", {})
                if rt.get("points"):
                    angle_r = self.detector.findAngle(img, *rt["points"], draw=False)
                    if angle_r is not None:
                        # Up position threshold
                        up_lo, up_hi = rt.get("up_position", [35, 55])
                        down_lo, down_hi = rt.get("down_position", [80, 100])

                        if up_lo <= angle_r <= up_hi and self.direction == 0:
                            self.direction = 1

                        if down_lo <= angle_r <= down_hi and self.direction == 1:
                            self.rep_count += 1
                            self.backend_text.insert('end', f"{name} Rep {self.rep_count}/{exercise.get('target_reps', 10)} (Set {self.current_set}/{exercise.get('sets', 3)})\n")
                            self.backend_text.see('end')
                            self.direction = 0

                        # Completed set?
                        if self.rep_count >= exercise.get("target_reps", 10):
                            self.backend_text.insert('end', f"Set {self.current_set}/{exercise.get('sets', 3)} complete for {name}\n")
                            self.backend_text.see('end')
                            self.rep_count = 0
                            self.direction = 0
                            self.current_set += 1

                            if self.current_set > exercise.get("sets", 3):
                                self.backend_text.insert('end', f"Completed {name}. Moving to next exercise.\n")
                                self.backend_text.see('end')
                                self.idx += 1
                                self.current_set = 1

                                if self.idx >= len(self.exercises):
                                    self.backend_text.insert('end', "\n🎉 Workout completed! Great job!\n")
                                    self.backend_text.see('end')
                                    self.stop_workout()
                                    return img
            elif etype == "time":
                tt = exercise.get("target_time", 30)
                holds = exercise.get("time_tracking", {}).get("hold_angles", [])
                good = True

                for ha in holds:
                    if ha.get("points"):
                        h_angle = self.detector.findAngle(img, *ha["points"], draw=False)
                        lo, hi = ha.get("required_range", [170, 190])
                        if h_angle is None or h_angle < lo or h_angle > hi:
                            self.backend_text.insert('end', f"Warning [{name}]: hold within {lo}-{hi}°\n")
                            self.backend_text.see('end')
                            good = False

                if good:
                    if self.start_time is None:
                        self.start_time = time.time()
                    else:
                        elapsed = time.time() - self.start_time
                        
                        self.backend_text.insert('end', f"{name} Hold {int(elapsed)}/{tt} sec\n")
                        self.backend_text.see('end')
                        if elapsed >= tt:
                            self.backend_text.insert('end', f"Set {self.current_set}/{exercise.get('sets', 3)} complete for {name}\n")
                            self.backend_text.see('end')
                            self.start_time = None
                            self.current_set += 1

                            if self.current_set > exercise.get("sets", 3):
                                self.backend_text.insert('end', f"Completed {name}. Moving to next exercise.\n")
                                self.backend_text.see('end')
                                self.idx += 1
                                self.current_set = 1

                                if self.idx >= len(self.exercises):
                                    self.stop_workout()
                                    return img
                else:
                    self.start_time = None

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - self.pTime) if self.pTime != 0 else 0
        self.pTime = cTime

        if(etype=='reps'):
            cv2.putText(img, f"{self.rep_count}", (20, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        else:
            cv2.putText(img,f"{elapsed}", (20, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        disp = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
        return disp     # returns the image frame.



if __name__ == "__main__":
    root = tk.Tk()        #Intializes the entire code
    app = WorkoutApp(root)
    root.mainloop()       #Mainloop that runs through all the functions