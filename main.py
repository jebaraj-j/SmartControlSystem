from ui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    try:
        # Create application
        app = QApplication(sys.argv)

        # Set application properties
        app.setApplicationName("Smart Control System")
        app.setOrganizationName("SmartControl")

        # Create and show main window
        window = MainWindow()
        window.show()

        print("Application started successfully!")
        print("Close the window to exit.")

        # Start event loop - this keeps the app running
        sys.exit(app.exec_())

    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback

        traceback.print_exc()
        input("Press Enter to exit...")