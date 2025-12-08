from abc import ABC, abstractmethod

from adaptive_router.models.storage import RouterProfile


class ProfileLoader(ABC):
    @abstractmethod
    def load_profile(self) -> RouterProfile:
        raise NotImplementedError

    def health_check(self) -> bool:
        return True
