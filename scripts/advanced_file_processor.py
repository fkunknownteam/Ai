import os
import json
import csv
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import zipfile
import shutil
from pathlib import Path
import hashlib
import mimetypes
from typing import Dict, List, Any, Optional
import time

class AdvancedFileProcessor:
    def __init__(self):
        self.processed_files = []
        self.file_stats = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive file analysis"""
        if not os.path.exists(file_path):
            return {'error': f'File not found: {file_path}'}
        
        try:
            file_info = {
                'path': file_path,
                'name': os.path.basename(file_path),
                'size': os.path.getsize(file_path),
                'size_human': self.format_file_size(os.path.getsize(file_path)),
                'extension': os.path.splitext(file_path)[1].lower(),
                'mime_type': mimetypes.guess_type(file_path)[0],
                'created': time.ctime(os.path.getctime(file_path)),
                'modified': time.ctime(os.path.getmtime(file_path)),
                'hash_md5': self.calculate_file_hash(file_path),
                'is_binary': self.is_binary_file(file_path)
            }
            
            # Add specific analysis based on file type
            if file_info['extension'] in ['.txt', '.py', '.js', '.html', '.css', '.md']:
                file_info.update(self.analyze_text_file(file_path))
            elif file_info['extension'] in ['.csv']:
                file_info.update(self.analyze_csv_file(file_path))
            elif file_info['extension'] in ['.json']:
                file_info.update(self.analyze_json_file(file_path))
            elif file_info['extension'] in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                file_info.update(self.analyze_image_file(file_path))
            
            return file_info
            
        except Exception as e:
            return {'error': f'Error analyzing file: {str(e)}'}
    
    def format_file_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return "Unable to calculate hash"
    
    def is_binary_file(self, file_path: str) -> bool:
        """Check if file is binary"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except:
            return True
    
    def analyze_text_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze text-based files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            words = content.split()
            
            return {
                'line_count': len(lines),
                'word_count': len(words),
                'character_count': len(content),
                'character_count_no_spaces': len(content.replace(' ', '')),
                'blank_lines': sum(1 for line in lines if not line.strip()),
                'average_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
                'encoding': 'utf-8'
            }
        except Exception as e:
            return {'text_analysis_error': str(e)}
    
    def analyze_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze CSV files"""
        try:
            df = pd.read_csv(file_path)
            
            return {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns)
            }
        except Exception as e:
            return {'csv_analysis_error': str(e)}
    
    def analyze_json_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            def count_elements(obj, depth=0):
                if isinstance(obj, dict):
                    return {
                        'type': 'object',
                        'keys': len(obj),
                        'depth': depth,
                        'nested_objects': sum(1 for v in obj.values() if isinstance(v, dict)),
                        'nested_arrays': sum(1 for v in obj.values() if isinstance(v, list))
                    }
                elif isinstance(obj, list):
                    return {
                        'type': 'array',
                        'length': len(obj),
                        'depth': depth,
                        'item_types': list(set(type(item).__name__ for item in obj))
                    }
                else:
                    return {'type': type(obj).__name__, 'depth': depth}
            
            structure = count_elements(data)
            
            return {
                'json_structure': structure,
                'is_valid_json': True,
                'root_type': type(data).__name__
            }
        except Exception as e:
            return {'json_analysis_error': str(e), 'is_valid_json': False}
    
    def analyze_image_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze image files"""
        try:
            with Image.open(file_path) as img:
                return {
                    'dimensions': img.size,
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                    'aspect_ratio': round(img.width / img.height, 2) if img.height > 0 else 0
                }
        except Exception as e:
            return {'image_analysis_error': str(e)}
    
    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """Process entire directory with comprehensive analysis"""
        if not os.path.exists(directory_path):
            return {'error': f'Directory not found: {directory_path}'}
        
        print(f"📁 Processing directory: {directory_path}")
        
        directory_stats = {
            'path': directory_path,
            'total_files': 0,
            'total_directories': 0,
            'total_size': 0,
            'file_types': {},
            'largest_files': [],
            'files_by_extension': {},
            'processing_time': time.time()
        }
        
        all_files = []
        
        # Walk through directory
        for root, dirs, files in os.walk(directory_path):
            directory_stats['total_directories'] += len(dirs)
            
            for file in files:
                file_path = os.path.join(root, file)
                file_info = self.analyze_file(file_path)
                
                if 'error' not in file_info:
                    all_files.append(file_info)
                    directory_stats['total_files'] += 1
                    directory_stats['total_size'] += file_info['size']
                    
                    # Track file extensions
                    ext = file_info['extension']
                    if ext not in directory_stats['files_by_extension']:
                        directory_stats['files_by_extension'][ext] = {'count': 0, 'total_size': 0}
                    directory_stats['files_by_extension'][ext]['count'] += 1
                    directory_stats['files_by_extension'][ext]['total_size'] += file_info['size']
        
        # Calculate statistics
        directory_stats['total_size_human'] = self.format_file_size(directory_stats['total_size'])
        directory_stats['average_file_size'] = directory_stats['total_size'] / directory_stats['total_files'] if directory_stats['total_files'] > 0 else 0
        directory_stats['processing_time'] = time.time() - directory_stats['processing_time']
        
        # Find largest files
        directory_stats['largest_files'] = sorted(all_files, key=lambda x: x['size'], reverse=True)[:10]
        
        # Most common file types
        directory_stats['most_common_extensions'] = sorted(
            directory_stats['files_by_extension'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:10]
        
        self.processed_files.extend(all_files)
        return directory_stats
    
    def create_sample_files(self) -> str:
        """Create sample files for demonstration"""
        sample_dir = "sample_files"
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create sample text file
        with open(f"{sample_dir}/sample.txt", 'w') as f:
            f.write("This is a sample text file.\nIt contains multiple lines.\nUsed for demonstration purposes.")
        
        # Create sample CSV file
        sample_data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'Age': [25, 30, 35, 28, 32],
            'City': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'],
            'Salary': [50000, 60000, 70000, 55000, 65000]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(f"{sample_dir}/sample.csv", index=False)
        
        # Create sample JSON file
        sample_json = {
            "users": [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
            ],
            "metadata": {
                "version": "1.0",
                "created": "2024-01-01"
            }
        }
        with open(f"{sample_dir}/sample.json", 'w') as f:
            json.dump(sample_json, f, indent=2)
        
        # Create sample Python file
        with open(f"{sample_dir}/sample.py", 'w') as f:
            f.write("""#!/usr/bin/env python3
def hello_world():
    print("Hello, World!")
    return "Success"

if __name__ == "__main__":
    hello_world()
""")
        
        print(f"📁 Sample files created in '{sample_dir}' directory")
        return sample_dir
    
    def generate_file_report(self, directory_stats: Dict[str, Any]) -> str:
        """Generate comprehensive file analysis report"""
        report = f"""
📊 FILE ANALYSIS REPORT
{'=' * 50}

📁 Directory: {directory_stats['path']}
⏱️ Processing Time: {directory_stats['processing_time']:.2f} seconds

📈 SUMMARY STATISTICS:
   Total Files: {directory_stats['total_files']:,}
   Total Directories: {directory_stats['total_directories']:,}
   Total Size: {directory_stats['total_size_human']}
   Average File Size: {self.format_file_size(directory_stats['average_file_size'])}

🏆 LARGEST FILES:
"""
        
        for i, file_info in enumerate(directory_stats['largest_files'][:5], 1):
            report += f"   {i}. {file_info['name']} - {file_info['size_human']}\n"
        
        report += f"""
📋 MOST COMMON FILE TYPES:
"""
        
        for ext, stats in directory_stats['most_common_extensions'][:10]:
            report += f"   {ext or 'No extension'}: {stats['count']} files ({self.format_file_size(stats['total_size'])})\n"
        
        return report
    
    def optimize_images(self, directory_path: str, quality: int = 85) -> Dict[str, Any]:
        """Optimize images in directory"""
        if not os.path.exists(directory_path):
            return {'error': f'Directory not found: {directory_path}'}
        
        optimized_dir = os.path.join(directory_path, 'optimized')
        os.makedirs(optimized_dir, exist_ok=True)
        
        results = {
            'processed': 0,
            'total_size_before': 0,
            'total_size_after': 0,
            'files': []
        }
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for root, dirs, files in os.walk(directory_path):
            if 'optimized' in root:  # Skip optimized directory
                continue
                
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    file_path = os.path.join(root, file)
                    
                    try:
                        original_size = os.path.getsize(file_path)
                        
                        with Image.open(file_path) as img:
                            # Convert RGBA to RGB if saving as JPEG
                            if img.mode == 'RGBA' and file.lower().endswith(('.jpg', '.jpeg')):
                                img = img.convert('RGB')
                            
                            # Optimize and save
                            optimized_path = os.path.join(optimized_dir, file)
                            img.save(optimized_path, optimize=True, quality=quality)
                            
                            optimized_size = os.path.getsize(optimized_path)
                            
                            results['files'].append({
                                'original': file_path,
                                'optimized': optimized_path,
                                'original_size': original_size,
                                'optimized_size': optimized_size,
                                'savings': original_size - optimized_size,
                                'savings_percent': ((original_size - optimized_size) / original_size) * 100
                            })
                            
                            results['processed'] += 1
                            results['total_size_before'] += original_size
                            results['total_size_after'] += optimized_size
                            
                    except Exception as e:
                        print(f"⚠️ Error optimizing {file}: {str(e)}")
        
        results['total_savings'] = results['total_size_before'] - results['total_size_after']
        results['total_savings_percent'] = (results['total_savings'] / results['total_size_before']) * 100 if results['total_size_before'] > 0 else 0
        
        return results

# Demonstration
if __name__ == "__main__":
    processor = AdvancedFileProcessor()
    
    print("📁 Advanced File Processing System")
    print("=" * 50)
    
    # Create sample files for demonstration
    sample_dir = processor.create_sample_files()
    
    # Process the sample directory
    print(f"\n🔍 Analyzing directory: {sample_dir}")
    directory_stats = processor.process_directory(sample_dir)
    
    # Generate and display report
    report = processor.generate_file_report(directory_stats)
    print(report)
    
    # Analyze individual files
    print("\n📄 INDIVIDUAL FILE ANALYSIS:")
    for file_info in processor.processed_files[:3]:  # Show first 3 files
        print(f"\n📋 {file_info['name']}:")
        print(f"   Size: {file_info['size_human']}")
        print(f"   Type: {file_info['mime_type']}")
        print(f"   Hash: {file_info['hash_md5'][:16]}...")
        
        # Show specific analysis based on file type
        if 'line_count' in file_info:
            print(f"   Lines: {file_info['line_count']}")
            print(f"   Words: {file_info['word_count']}")
        elif 'rows' in file_info:
            print(f"   Rows: {file_info['rows']}")
            print(f"   Columns: {file_info['columns']}")
        elif 'dimensions' in file_info:
            print(f"   Dimensions: {file_info['dimensions']}")
    
    print("\n✅ File processing demonstration completed!")
    print(f"📊 Total files processed: {len(processor.processed_files)}")
    print(f"💾 Total size analyzed: {processor.format_file_size(sum(f['size'] for f in processor.processed_files))}")
