import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Canvas, Frame, Scrollbar,simpledialog
import numpy as np
import cv2
from PIL import Image, ImageTk

import project

class ImageProcessingApp:
    def __init__(self, master):
        super().__init__()
        self.original_image = None
        self.processed_image = None

        self.master = master
        master.title("Image Processing Application")
        master.geometry("1000x700")
        master.configure(bg='#2E3B4E')

        # Original and Processed Image Frames
        image_frame = tk.Frame(master, bg='#2E3B4E')
        image_frame.pack(pady=10, padx=10)

        # Original Image Display
        self.original_label = tk.Label(image_frame, text="Original Image", bg='#2E3B4E', fg='white', font=("Helvetica", 14))
        self.original_label.grid(row=0, column=0, padx=20)
        self.original_image_display = tk.Label(image_frame, bg='#2E3B4E')
        self.original_image_display.grid(row=1, column=0, padx=20)

        # Processed Image Display
        self.processed_label = tk.Label(image_frame, text="Processed Image", bg='#2E3B4E', fg='white', font=("Helvetica", 14))
        self.processed_label.grid(row=0, column=1, padx=20)
        self.processed_image_display = tk.Label(image_frame, bg='#2E3B4E')
        self.processed_image_display.grid(row=1, column=1, padx=20)

        # Control Frame
        control_frame = tk.Frame(master, bg='#2E3B4E')
        control_frame.pack(pady=10)

        # Load and Save Buttons
        load_button = tk.Button(control_frame, text="Load Image", command=self.load_image, bg='#4CAF50', fg='white', font=("Helvetica", 12), width=15)
        load_button.grid(row=0, column=0, padx=10, pady=5)
        save_button = tk.Button(control_frame, text="Save Result", command=self.save_image, bg='#F44336', fg='white', font=("Helvetica", 12), width=15)
        save_button.grid(row=0, column=1, padx=10, pady=5)

        # Category Selection
        category_frame = tk.Frame(master, bg='#2E3B4E')
        category_frame.pack(pady=10)
        tk.Label(category_frame, text="Select Category:", bg='#2E3B4E', fg='white', font=("Helvetica", 12)).grid(row=0, column=0, padx=10)
        self.category_var = tk.StringVar()
        self.category_dropdown = ttk.Combobox(category_frame, textvariable=self.category_var, state="readonly",
                                              values=list(self.operations.keys()), font=("Helvetica", 12), width=25)
        self.category_dropdown.grid(row=0, column=1, padx=10)
        self.category_dropdown.bind("<<ComboboxSelected>>", self.update_buttons)

        # Operations Frame
        self.operations_frame = tk.Frame(master, bg='#2E3B4E')
        self.operations_frame.pack(pady=10, padx=10, expand=True)


    # Define Operations by Category
    operations = {
        "Image Color": ["Grayscale"],
        "Thresholding": ["Threshold"],
        "Halftoning": ["Simple Halftone Threshold", "Advanced Halftone Error Diffusion"],
        "Histogram Processing": ["Histogram"],
        "Simple Edge Detection": ["Sobel Operator", "Prewitt Operator", "Kirsch Compass Masks"],
        "Advanced Edge Detection": ["Homogeneity Operator", "Difference Operator", "Difference Of Gaussians Operator",
                                     "Contrast-based Edge Detection-Smoothing Mask", "Variance Mask", "Range Mask"],
        "Filtering": ["High-pass Filter", "Low-pass Filter", "Median Filter"],
        "Image Operations": ["Add To Copy Of Image", "Subtract To Copy Of Image", "Invert Image"],
        "Histogram Based Segmentation": ["Manual Histogram Segmentation", "Histogram Peak Technique",
                                          "Histogram Valley Technique", "Adaptive Histogram Technique"]
    }

    def update_buttons(self, event=None):
        # Clear current buttons
        for widget in self.operations_frame.winfo_children():
            widget.destroy()

        # Add buttons for selected category
        selected_category = self.category_var.get()
        if selected_category in self.operations:
            operations = self.operations[selected_category]
            for i, operation in enumerate(operations):
                button = tk.Button(self.operations_frame, text=operation,
                                command=lambda op=operation: self.process_image(op),
                                bg='#2196F3', fg='white', font=("Helvetica", 10), width=25, height=2)
                button.grid(row=i // 4, column=i % 4, padx=10, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            messagebox.showinfo("Info", "No file selected.")
            print("File dialog canceled.")
            return
        try:
            image = Image.open(file_path)
            self.original_image = np.array(image)  # Convert PIL Image to numpy array
            print(f"Image loaded successfully: {self.original_image.shape}")
            self.display_image(self.original_image, self.original_image_display)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def process_image(self, operation):
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            print("Error: Original image is None.")
            return

        try:
            grayscale_array = project.prepare_grayscale_array(self.original_image)
            # Perform operations based on the selection
            if operation == "Grayscale":
                self.processed_image = project.convert_to_grayscale(grayscale_array)                
            elif operation == "Threshold":
                self.processed_image = project.apply_threshold(grayscale_array)
            elif operation == "Simple Halftone Threshold":
                self.processed_image = project.simple_halftoning(grayscale_array)
            elif operation == "Advanced Halftone Error Diffusion":
                self.processed_image = project.Advanced_halftoning(grayscale_array)
            elif operation == "Histogram":
                self.processed_image, _, _, _ = project.histogram_equalization(grayscale_array)
            elif operation == "Sobel Operator":
                self.processed_image = project.sobel_operator(grayscale_array)
            elif operation == "Prewitt Operator":
                self.processed_image = project.prewitt_operator(grayscale_array)
            elif operation == "Kirsch Compass Masks":
                kirsch_edges, _ = project.kirsch_compass_masks(grayscale_array)
                self.processed_image = kirsch_edges
            elif operation == "Homogeneity Operator":
                self.processed_image = project.homogeneity_operator(grayscale_array)
            elif operation == "Difference Operator":                
                self.processed_image = project.difference_operator(grayscale_array)
            elif operation == "Difference Of Gaussians Operator":                
                self.processed_image,smoothed1,smoothed2 = project.difference_of_guassians(grayscale_array, project.kernel7, project.kernel9)    
            elif operation == "Contrast-based Edge Detection-Smoothing Mask":                
                self.processed_image = project.contrast_based_edge_detection_with_edge_detector(grayscale_array)         
            elif operation == "Variance Mask":                
                self.processed_image = project.variance_edge_detector(grayscale_array)
            elif operation == "Range Mask":                
                self.processed_image = project.range_edge_detector(grayscale_array)    
            elif operation == "High-pass Filter":                
                self.processed_image = project.high_pass_filter(grayscale_array)    
            elif operation == "Low-pass Filter":                
                self.processed_image = project.low_pass_filter(grayscale_array)
            elif operation == "Median Filter":                
                self.processed_image = project.median_filter(grayscale_array)
            elif operation == "Add To Copy Of Image":                
                self.processed_image = project.add_imagess(grayscale_array)    
            elif operation == "Subtract To Copy Of Image":                
                self.processed_image = project.subtract_imagess(grayscale_array)    
            elif operation == "Invert Image":                
                self.processed_image = project.invert_image(grayscale_array)             
            elif operation == "Manual Histogram Segmentation":
                while True:  # Keep asking until valid thresholds are provided
                    thresholds = simpledialog.askstring("Input", "Enter thresholds (comma-separated, between 0 and 255):")
                    if not thresholds:
                        messagebox.showwarning("Warning", "Input cannot be empty!")
                        continue

                    try:
                        # Parse and validate the input
                        thresholds = list(map(int, thresholds.split(',')))
                        if all(0 <= t <= 255 for t in thresholds):
                            break  # Exit loop if valid thresholds are provided
                        else:
                            messagebox.showerror("Error", "All thresholds must be between 0 and 255.")
                    except ValueError:
                        messagebox.showerror("Error", "Invalid input! Please enter integers only.")

                self.processed_image = project.manual_histogram_segmentation(self.original_image, thresholds)

            elif operation == "Histogram Peak Technique":
                self.processed_image, _, _, _, _ = project.histogram_peak_thresholding(self.original_image)
            elif operation == "Histogram Valley Technique":
                segmented_image, histogram, peaks, valley = project.valley_based_segmentation(grayscale_array)
                self.processed_image = segmented_image                  
            elif operation == "Adaptive Histogram Technique":
                result = project.adaptive_histogram_segmentation(grayscale_array)
                self.processed_image = result["final_segmented_image"]                
            else:
                messagebox.showerror("Error", f"Operation '{operation}' not implemented.")

            # Display processed image
            if self.processed_image is not None:
                print(f"Operation '{operation}' successful. Displaying result.")
                self.display_image(self.processed_image, self.processed_image_display)
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            print(f"Processing failed for operation '{operation}': {str(e)}")
            
    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "No processed image to save!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            Image.fromarray(self.processed_image).save(file_path)
            messagebox.showinfo("Success", "Image saved successfully!")

    def display_image(self, image, label):
        image = Image.fromarray(image).resize((300, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(image)
        label.img_tk = img_tk  # Prevent garbage collection
        label.config(image=img_tk)

def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()