import os
import zipfile
from pathlib import Path

def zip_dataset(source_dir, output_zip, process_level):
    """
    Zip the dataset according to specific requirements:
    - For non-neutral emotions: only include level_2
    - For neutral emotion: include level_1
    - Skip level_3 and level_1 for non-neutral emotions
    - Include both video/front and audio directories
    """
    # Create a zip file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the source directory
        for root, dirs, files in os.walk(source_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            # Get the relative path from source directory
            rel_path = os.path.relpath(root, source_dir)
            
            # Check if we're in a valid directory structure
            path_parts = Path(rel_path).parts
            if len(path_parts) < 4:  # Not deep enough in the directory structure
                continue
                
            subject, data_type, front, emotion = path_parts[:4]
            
            # Skip if not in video/front or audio directory
            if data_type not in ['video', 'audio']:
                continue
                
            if data_type == 'video' and front != 'front':
                continue

            # Check emotion and level
            if len(path_parts) > 4:  # We have a level
                level = path_parts[4]

                # Skip if:
                # - Neutral emotion and not level_1
                # - Non-neutral emotion and not process_level
                if emotion != 'neutral' and level != process_level:
                    continue
                if emotion == 'neutral' and level != 'level_1':
                    continue
            
            # Add files to zip
            for file in files:
                if not file.startswith('.'):  # Skip hidden files
                    file_path = os.path.join(root, file)
                    print(file_path)
                    arcname = os.path.join(rel_path, file)
                    zipf.write(file_path, arcname)

if __name__ == "__main__":
    # Example usage
    source_directory = '/Users/xiaokeai/Downloads/MEAD/'  # Adjust this path as needed
    output_zip_file = '/Users/xiaokeai/Downloads/MEAD_zipped.zip'
    process_level = 'level_2'
    
    print(f"Starting to zip dataset from {source_directory}...")
    zip_dataset(source_directory, output_zip_file, process_level)
    print(f"Dataset has been zipped to {output_zip_file}")
