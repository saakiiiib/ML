import os

file_path = CONFIG["dataset_path"]

print(f"Inspecting file: {file_path}")

# Read first few bytes to check for magic numbers (e.g., for zip/xlsx)
with open(file_path, 'rb') as f:
    first_bytes = f.read(4)
    print(f"First 4 bytes (hex): {first_bytes.hex()}")
    if first_bytes == b'PK\x03\x04':
        print("\nThis file appears to be a ZIP archive or an Excel XLSX file.")
        print("Please save your data as a plain CSV (e.g., UTF-8 encoded, comma-delimited) and re-upload it.")
    else:
        print("\nFile does not appear to be a ZIP/XLSX archive based on magic number.")

# Try to read first few lines as text to inspect structure
print("\n--- First 10 lines of the file (as text) ---")
try:
    with open(file_path, 'r', encoding='latin1') as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i >= 9: # Print first 10 lines
                break
except UnicodeDecodeError:
    print("Could not decode file with latin1. It might be binary or a different encoding.")
except Exception as e:
    print(f"Error reading file as text: {e}")

print("\n--- Attempting to read with various delimiters for diagnosis ---")
# Diagnostic attempts with different delimiters if it's indeed a text file
def try_read_csv(filepath, sep=None, encoding='latin1'):
    try:
        df_test = pd.read_csv(filepath, sep=sep, encoding=encoding, on_bad_lines='warn', nrows=5)
        print(f"Successfully read with sep='{sep}' and encoding='{encoding}'. Head:\n{df_test.head()}")
        return True
    except pd.errors.ParserError as e:
        print(f"Failed with sep='{sep}' and encoding='{encoding}': ParserError: {e}")
    except UnicodeDecodeError as e:
        print(f"Failed with sep='{sep}' and encoding='{encoding}': UnicodeDecodeError: {e}")
    except Exception as e:
        print(f"Failed with sep='{sep}' and encoding='{encoding}': {type(e).__name__}: {e}")
    return False

print("\nTrying comma delimiter:")
try_read_csv(file_path, sep=',')

print("\nTrying semicolon delimiter:")
try_read_csv(file_path, sep=';')

print("\nTrying tab delimiter:")
try_read_csv(file_path, sep='\t')

print("\nTrying whitespace delimiter (r'\\s+'):")
try_read_csv(file_path, sep=r'\s+')
