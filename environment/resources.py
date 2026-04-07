"""
Resource Management Module

Manages hospital resources including ICU beds, equipment, and facilities.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ResourceStatus(Enum):
    """Enumeration of possible resource statuses."""
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
    MAINTENANCE = "maintenance"
    OUT_OF_ORDER = "out_of_order"


@dataclass
class ICUBed:
    """Represents an ICU bed resource."""
    bed_id: str
    status: ResourceStatus
    current_patient_id: Optional[str] = None
    equipment_list: List[str] = None
    
    def __post_init__(self):
        if self.equipment_list is None:
            self.equipment_list = []


@dataclass
class Equipment:
    """Represents medical equipment resource."""
    equipment_id: str
    equipment_type: str
    status: ResourceStatus
    location: Optional[str] = None


class ResourceManager:
    """
    Manages all hospital resources.
    
    Attributes:
        icu_beds: Dictionary of ICU beds by ID
        equipment: Dictionary of equipment by ID
        total_capacity: Total resource capacity
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize resource manager.
        
        Args:
            config: Configuration for resource initialization
        """
        self.config = config
        self.icu_beds: Dict[str, ICUBed] = {}
        self.equipment: Dict[str, Equipment] = {}
        self._initialize_resources()
    
    def _initialize_resources(self) -> None:
        """Initialize resources based on configuration."""
        cfg_resources = self.config.get("resources", self.config)

        bed_count = int(cfg_resources.get("beds", cfg_resources.get("icu_beds", 0)))
        equipment_types: List[str] = list(
            cfg_resources.get("equipment_types", ["monitor", "ventilator", "infusion_pump"])
        )

        for i in range(max(0, bed_count)):
            bed_id = f"bed_{i}"
            self.icu_beds[bed_id] = ICUBed(
                bed_id=bed_id,
                status=ResourceStatus.AVAILABLE,
                current_patient_id=None,
                equipment_list=list(cfg_resources.get("bed_equipment", ["monitor"])),
            )

        equipment_count = int(cfg_resources.get("equipment_count", max(1, bed_count * 2)))
        for i in range(max(0, equipment_count)):
            eq_type = equipment_types[i % len(equipment_types)]
            eq_id = f"equipment_{i}"
            self.equipment[eq_id] = Equipment(
                equipment_id=eq_id,
                equipment_type=eq_type,
                status=ResourceStatus.AVAILABLE,
                location=None,
            )
    
    def allocate_bed(self, bed_id: str, patient_id: str) -> bool:
        """
        Allocate a bed to a patient.
        
        Args:
            bed_id: ID of the bed to allocate
            patient_id: ID of the patient
            
        Returns:
            True if allocation successful, False otherwise
        """
        bed = self.icu_beds.get(bed_id)
        if bed is None:
            return False
        if bed.status != ResourceStatus.AVAILABLE:
            return False
        bed.status = ResourceStatus.OCCUPIED
        bed.current_patient_id = patient_id
        return True
    
    def release_bed(self, bed_id: str) -> bool:
        """
        Release a bed from its current patient.
        
        Args:
            bed_id: ID of the bed to release
            
        Returns:
            True if release successful, False otherwise
        """
        bed = self.icu_beds.get(bed_id)
        if bed is None:
            return False
        bed.status = ResourceStatus.AVAILABLE
        bed.current_patient_id = None
        return True
    
    def get_available_beds(self) -> List[ICUBed]:
        """
        Get list of currently available beds.
        
        Returns:
            List of available ICU beds
        """
        return [bed for bed in self.icu_beds.values() if bed.status == ResourceStatus.AVAILABLE]
    
    def get_utilization_stats(self) -> Dict[str, float]:
        """
        Get resource utilization statistics.
        
        Returns:
            Dictionary with utilization metrics
        """
        total_beds = len(self.icu_beds)
        occupied = sum(1 for b in self.icu_beds.values() if b.status == ResourceStatus.OCCUPIED)
        available = sum(1 for b in self.icu_beds.values() if b.status == ResourceStatus.AVAILABLE)
        reserved = sum(1 for b in self.icu_beds.values() if b.status == ResourceStatus.RESERVED)
        maintenance = sum(1 for b in self.icu_beds.values() if b.status in {ResourceStatus.MAINTENANCE, ResourceStatus.OUT_OF_ORDER})

        bed_utilization = (occupied / total_beds) if total_beds > 0 else 0.0

        total_equipment = len(self.equipment)
        equipment_available = sum(1 for e in self.equipment.values() if e.status == ResourceStatus.AVAILABLE)
        equipment_utilization = 1.0 - ((equipment_available / total_equipment) if total_equipment > 0 else 1.0)

        return {
            "total_beds": float(total_beds),
            "occupied_beds": float(occupied),
            "available_beds": float(available),
            "reserved_beds": float(reserved),
            "maintenance_beds": float(maintenance),
            "bed_utilization": float(bed_utilization),
            "total_equipment": float(total_equipment),
            "equipment_utilization": float(max(0.0, equipment_utilization)),
        }
    
    def update_resource_status(self, resource_id: str, status: ResourceStatus) -> bool:
        """
        Update status of a resource.
        
        Args:
            resource_id: ID of the resource
            status: New status to set
            
        Returns:
            True if update successful, False otherwise
        """
        if resource_id in self.icu_beds:
            bed = self.icu_beds[resource_id]
            bed.status = status
            if status != ResourceStatus.OCCUPIED:
                bed.current_patient_id = None
            return True

        if resource_id in self.equipment:
            self.equipment[resource_id].status = status
            return True

        return False
