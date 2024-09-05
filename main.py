from screenshot_processor import process_screenshots

def main():
    print("===== ENTERING main FUNCTION =====")
    process_screenshots()
    print("===== EXITING main FUNCTION =====")

if __name__ == "__main__":
    print("===== SCRIPT STARTED =====")
    try:
        main()
    except ImportError as e:
        print(f"===== IMPORT ERROR: {e} =====")
    print("===== SCRIPT ENDED =====")

    # Keep console open
    input("Press Enter to exit...")
