from abc import ABC, abstractmethod

class Report(ABC):
    @abstractmethod
    def generate_streamlit_report(self, *args):
        pass
