###########################################################################
#Simulation of sortation center processes with a probabilistic demand arrival
# Description: This file contains the simulation model for the dual sortation center simulation.
# By: Andrew Fenstermacher
# This file is used to show the recommended process change to have Line Haul C Pallets discrimintated into TLMD and Not TLMD at the store before arrival. Should be used
# in conjunction with the Dual_sim_SC.py file to generate a comparison data set.

import math
import numpy as np
import sim_generator_LHC_Norm_test as sg_c
import matplotlib.pyplot as plt
import pandas as pd
import simpy
import gc
import random

class G:
    # Constants (adjust as needed)
    #Process_Variance = 0
    UNLOADING_RATE = 60/45  # minutes per pallet
    FLUID_UNLOAD_RATE = 60/600  # minutes per package
    INDUCT_STAGE_RATE = 60/45  # minutes per pallet
    INDUCTION_RATE = 60/700  # minutes per package
    SPLITTER_RATE = 60/3600 #60/1049  # minutes per package
    TLMD_BUFFER_SORT_RATE = 60/500  # minutes per package
    TLMD_PARTITION_STAGE_RATE = 60/13  # minutes per pallet
    TLMD_INDUCT_STAGE_RATE = 60/30  # minutes per pallet
    TLMD_INDUCTION_RATE = 60/330  # minutes per package
    TLMD_FINAL_SORT_RATE = 60/130  # minutes per package
    TLMD_CART_STAGE_RATE = 60/105  # minutes per cart
    TLMD_CART_HANDOFF_RATE = 60/22  # minutes per pallet
    CART_STAGE_RATE = 2  # minutes per cart
    TLMD_CART_HANDOFF_RATE = 2  # minutes per cart
    NATIONAL_CARRIER_SORT_RATE = 60/165  # minutes per package
    NC_PALLET_STAGING_RATE = 60/120  # minutes per pallet
    NATIONAL_CARRIER_FLUID_PICK_RATE = 1/60  # minutes per package
    NATIONAL_CARRIER_FLUID_LOAD_RATE = 60/120  # minutes per package
    TLMD_C_PARTITION_STAGE_RATE = 60/15
    


    OUTBOUND_NC_PALLET_MAX_PACKAGES = 50  # Max packages per pallet
    PARTITION_PALLET_MAX_PACKAGES = 65  # Max packages per pallet
    TLMD_PARTITION_PALLET_MAX_PACKAGES = 65  # Max packages per pallet
    NC_PALLET_MAX_PACKAGES = 50  # Max packages per pallet
    TLMD_CART_MAX_PACKAGES = 20  # Max packages per cart
    TOTAL_PACKAGES = None  # Total packages to be processed
    TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
    TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
    TLMD_AB_INDUCT_TIME = None
    TLMD_C_INDUCT_TIME = None
    TLMD_STAGED_PACKAGES = None
    TLMD_PARTITION_1_PACKAGES = None
    TLMD_PARTITION_2_PACKAGES = None
    TLMD_PARTITION_3AB_PACKAGES = None
    TLMD_PARTITION_3C_PACKAGES = None
    TLMD_PARTITION_1_PALLETS = None
    TLMD_PARTITION_2_PALLETS = None
    TLMD_PARTITION_3AB_PALLETS = None
    TLMD_PARTITION_3C_PALLETS = None
    TOTAL_PALLETS_TLMD = None
    TLMD_PARTITION_1_SORT_TIME = None
    TLMD_PARTITION_2_SORT_TIME = None 
    TLMD_PARTITION_3AB_SORT_TIME = None
    TLMD_PARTITION_3_SORT_TIME = None
    TLMD_SORTED_PACKAGES = None
    TLMD_PARTITION_1_CART_STAGE_TIME = None
    TLMD_PARTITION_2_CART_STAGE_TIME = None 
    TLMD_PARTITION_3_CART_STAGE_TIME = None
    TLMD_OUTBOUND_PACKAGES = None
    I=1
    J=1
    K=1
    L=1
    M=1
    N=1

    PASSED_OVER_PACKAGES = None

    TOTAL_LINEHAUL_A_PACKAGES = None
    TOTAL_LINEHAUL_B_PACKAGES = None
    TOTAL_LINEHAUL_C_PACKAGES = None

    USPS_LINEHAUL_A_PACKAGES = None
    USPS_LINEHAUL_B_PACKAGES = None
    USPS_LINEHAUL_C_PACKAGES = None

    UPSN_LINEHAUL_A_PACKAGES = None
    UPSN_LINEHAUL_B_PACKAGES = None
    UPSN_LINEHAUL_C_PACKAGES = None

    FDEG_LINEHAUL_A_PACKAGES = None
    FDEG_LINEHAUL_B_PACKAGES = None
    FDEG_LINEHAUL_C_PACKAGES = None

    FDE_LINEHAUL_A_PACKAGES = None
    FDE_LINEHAUL_B_PACKAGES = None
    FDE_LINEHAUL_C_PACKAGES = None

    TLMD_LINEHAUL_A_PACKAGES = None
    TLMD_LINEHAUL_B_PACKAGES = None
    TLMD_LINEHAUL_C_PACKAGES = None

    TLMD_LINEHAUL_TFC_PACKAGES=None

    LINEHAUL_C_TIME = None
    LINEHAUL_TFC_TIME = None

    TOTAL_PACKAGES_UPSN = None
    TOTAL_PACKAGES_USPS = None
    TOTAL_PACKAGES_FDEG = None
    TOTAL_PACKAGES_FDE = None
    UPSN_PALLETS = None
    USPS_PALLETS = None
    FDEG_PALLETS = None
    FDE_PALLETS = None
    UPSN_SORT_TIME = None
    USPS_SORT_TIME = None
    FDEG_SORT_TIME = None
    FDE_SORT_TIME = None

    TOTAL_CARTS_TLMD = None

    PARTITION_1_RATIO = 0.50  # Ratio of packages to go to partition 1
    PARTITION_2_RATIO = 0.35  # Ratio of packages to go to partition 2
    PARTITION_3_RATIO = 0.15  # Ratio of packages to go to partition 3

class Package:
    def __init__(self, tracking_number, pallet_id, scac, partition):
        self.tracking_number = tracking_number
        self.pallet_id = pallet_id
        self.scac = scac
        self.partition = partition
        self.received_ts = None
        self.inducted_ts = None
        self.sorted_ts = None
        self.current_queue = None

class Pallet:
    def __init__(self, env, pallet_id, LH, packages, pkg_received_utc_ts):
        self.env = env
        self.pkg_received_utc_ts = pkg_received_utc_ts
        self.pallet_id = pallet_id
        self.LH = LH
        self.packages = [Package(pkg[0], pallet_id, pkg[1], pkg[2]) for pkg in packages]
        self.current_queue = None
        self.current_packages = len(packages)  # Track remaining packages

class TLMD_Pallet:
    def __init__(self, env, pallet_id, packages, built_time):
        self.env = env
        self.built_time = built_time
        self.pallet_id = pallet_id
        self.packages = packages
        self.current_packages = len(packages)  # Track remaining packages

    def add_package(self, package):
        self.packages.append(package)
        self.current_packages = len(self.packages)  # Update the count
        #print(f'Package {package.tracking_number} added to pallet {self.pallet_id} at {self.env.now}')
        #print(f'Pallet currently contains {self.current_packages} packages')

class TLMD_Cart:
    def __init__(self, env, cart_id, packages, built_time):
        self.env = env
        self.built_time = built_time
        self.cart_id = cart_id
        self.packages = packages
        self.current_packages = len(packages)  # Tracks packages
    
    def add_package(self, package):
        self.packages.append(package)
        self.current_packages = len(self.packages)  # Update the count
        #print(f"Package {package.tracking_number} added to cart {self.cart_id} at {self.env.now}")
        ##print(f'Cart currently contains {self.current_packages} packages')

class National_Carrier_Pallet:
    def __init__(self, env, pallet_id, packages, scac, built_time):
        self.env = env
        self.built_time = built_time
        self.pallet_id = pallet_id
        self.packages = packages
        self.scac = scac
        self.current_packages = len(packages)  # Track remaining packages

    def add_package(self, package):
        self.packages.append(package)
        self.current_packages = len(self.packages)  # Update the count
        #print(f"Package {package.tracking_number} added to pallet {self.pallet_id} at {self.env.now}")
        ##print(f'Pallet currently contains {self.current_packages} packages')

def manage_resources(env, sortation_center, current_resource, 
                     night_tm_pit_unload, 
                        night_tm_pit_induct, 
                        night_tm_nonpit_split, 
                        night_tm_nonpit_NC, 
                        night_tm_nonpit_buffer,
                        night_tm_TLMD_induct,
                        night_tm_TLMD_induct_stage,
                        night_tm_TLMD_picker,
                        night_tm_TLMD_sort,
                        night_tm_TLMD_stage,
                        day_tm_pit_unload,
                        day_tm_pit_induct,
                        day_tm_nonpit_split,
                        day_tm_nonpit_NC,
                        day_tm_nonpit_buffer,
                        day_tm_TLMD_induct,
                        day_tm_TLMD_induct_stage,
                        day_tm_TLMD_picker,
                        day_tm_TLMD_sort,
                        day_tm_TLMD_stage):
    
        yield env.timeout(30)
        # Start with 1 resource for the first 10 minutes
        sortation_center.current_resource['tm_pit_unload'] = simpy.Resource(env, capacity=night_tm_pit_unload)
        sortation_center.current_resource['tm_pit_induct_stage'] = simpy.Resource(env, capacity=1)
        sortation_center.current_resource['tm_pit_induct'] = simpy.PriorityResource(env, capacity=night_tm_pit_induct)
        sortation_center.current_resource['tm_nonpit_split'] = simpy.Resource(env, capacity=night_tm_nonpit_split)
        sortation_center.current_resource['tm_nonpit_NC'] = simpy.PriorityResource(env, capacity=night_tm_nonpit_NC)
        sortation_center.current_resource['tm_nonpit_buffer'] = simpy.PriorityResource(env, capacity=night_tm_nonpit_buffer)
        sortation_center.current_resource['tm_TLMD_induct'] = simpy.PriorityResource(env, capacity=night_tm_TLMD_induct)
        sortation_center.current_resource['tm_TLMD_induct_stage'] = simpy.PriorityResource(env, capacity=night_tm_TLMD_induct_stage)
        sortation_center.current_resource['tm_TLMD_picker'] = simpy.Resource(env, capacity=night_tm_TLMD_picker)
        sortation_center.current_resource['tm_TLMD_sort'] = simpy.Resource(env, capacity=night_tm_TLMD_sort)
        sortation_center.current_resource['tm_TLMD_stage'] = simpy.Resource(env, capacity=night_tm_TLMD_stage)
        sortation_center.current_resource['tm_TFC_unload'] = simpy.PriorityResource(env, capacity=night_tm_pit_unload +night_tm_pit_induct)
        sortation_center.current_resource['tm_TFC_sort'] = simpy.Resource(env, capacity=night_tm_nonpit_split + night_tm_nonpit_NC + night_tm_nonpit_buffer)

        #print(f"Using nightshift resources at time {env.now}")
        yield env.timeout(790)

        # Switch to 5 resources for the next 30 minutes
        sortation_center.current_resource['tm_pit_unload'] = simpy.Resource(env, capacity=day_tm_pit_unload)
        sortation_center.current_resource['tm_pit_induct'] = simpy.PriorityResource(env, capacity=day_tm_pit_induct)
        sortation_center.current_resource['tm_pit_induct_stage'] = simpy.Resource(env, capacity=1)
        sortation_center.current_resource['tm_nonpit_split'] = simpy.Resource(env, capacity=day_tm_nonpit_split)
        sortation_center.current_resource['tm_nonpit_NC'] = simpy.PriorityResource(env, capacity=day_tm_nonpit_NC)
        sortation_center.current_resource['tm_nonpit_buffer'] = simpy.PriorityResource(env, capacity=day_tm_nonpit_buffer)
        sortation_center.current_resource['tm_TLMD_induct'] = simpy.PriorityResource(env, capacity=day_tm_TLMD_induct)
        sortation_center.current_resource['tm_TLMD_induct_stage'] = simpy.PriorityResource(env, capacity=day_tm_TLMD_induct_stage)
        sortation_center.current_resource['tm_TLMD_picker'] = simpy.Resource(env, capacity=day_tm_TLMD_picker)
        sortation_center.current_resource['tm_TLMD_sort'] = simpy.Resource(env, capacity=day_tm_TLMD_sort)
        sortation_center.current_resource['tm_TLMD_stage'] = simpy.Resource(env, capacity=day_tm_TLMD_stage)
        sortation_center.current_resource['tm_TFC_unload'] = simpy.PriorityResource(env, capacity=day_tm_pit_unload +day_tm_pit_induct)
        sortation_center.current_resource['tm_TFC_sort'] = simpy.Resource(env, capacity=day_tm_nonpit_split + day_tm_nonpit_NC + day_tm_nonpit_buffer)

def break_preshift(env, sortation_center):
    yield env.timeout(0) 
    #print(f'Pausing all processes at {env.now}')
    sortation_center.pause_event = True  
    yield env.timeout(15)  
    sortation_center.pause_event = False 
    print(f'Resuming all processes at {env.now}')

def break_1(env, sortation_center):
    yield env.timeout(180) 
    #print(f'Pausing all processes at {env.now}')
    sortation_center.pause_event = True  
    yield env.timeout(30)  
    sortation_center.pause_event = False 
    #print(f'Resuming all processes at {env.now}')

def break_2(env, sortation_center):
    yield env.timeout(420)  
    #print(f'Pausing all processes at {env.now}')
    sortation_center.pause_event = True  
    yield env.timeout(15)  
    sortation_center.pause_event = False 
    #print(f'Resuming all processes at {env.now}')

def between_shift(env, sortation_center):
    yield env.timeout(660)  
    #print(f'Pausing all processes at {env.now}')
    sortation_center.pause_event = True  
    yield env.timeout(180)  
    sortation_center.pause_event = False 
    #print(f'Resuming all processes at {env.now}')

def break_3(env, sortation_center):
    yield env.timeout(960)  
    #print(f'Pausing all processes at {env.now}')
    sortation_center.pause_event = True  
    yield env.timeout(30)  
    sortation_center.pause_event = False 
    #print(f'Resuming all processes at {env.now}')

def break_4(env, sortation_center):
    yield env.timeout(1200)  
    #print(f'Pausing all processes at {env.now}')
    sortation_center.pause_event = True  
    yield env.timeout(15)  
    sortation_center.pause_event = False
    #print(f'Resuming all processes at {env.now}')

def linehaul_C_arrival(env, sortation_center):
    yield env.timeout(G.LINEHAUL_C_TIME)
    #print(f'Linehaul C arrival at {env.now}')
    sortation_center.LHC_flag = True
    sortation_center.LHC_arrive_flag = True

def TFC_arrival(env, sortation_center):
    yield env.timeout(G.LINEHAUL_TFC_TIME)
    #print(f'Linehaul C arrival at {env.now}')
    sortation_center.TFC_flag = True

def plot_metrics(metrics):
    plt.figure(figsize=(12, 8))
    plt.plot(metrics['resource_utilization'])
    plt.title('util')
    plt.xlabel('Time')
    plt.ylabel('util')
    plt.legend()
    plt.show()

    for queue, lengths in metrics['queue_lengths'].items():
        plt.figure(figsize=(12, 8))
        plt.plot(lengths, label=queue)
        plt.title('Queue Length')
        plt.xlabel('Time')
        plt.ylabel('Queue Length')
        plt.legend()
        plt.show()

def packages_to_dataframe(packages):
    data = {
        'tracking_number': [pkg.tracking_number for pkg in packages],
        'pallet_id': [pkg.pallet_id for pkg in packages],
        'scac': [pkg.scac for pkg in packages],
        'partition': [pkg.partition for pkg in packages],
        'received_ts': [pkg.received_ts for pkg in packages],
        'inducted_ts': [pkg.inducted_ts for pkg in packages],
        'sorted_ts': [pkg.sorted_ts for pkg in packages],
        'current_queue': [pkg.current_queue for pkg in packages],
    }
    return pd.DataFrame(data)

###################################################################################
class Sortation_Center_Original:
    def __init__(self, 
                env, 
                pallets_df, 
                USPS_Fluid_Status,
                UPSN_Fluid_Status,
                FDEG_Fluid_Status,
                FDE_Fluid_Status,
                current_resources,
                var_status,
                var_mag,
                TFC_status
                ):
        
        self.env = env
        self.pallets_df = pallets_df
        self.pause_event = False
        self.current_resource = current_resources

        #Determine whether National Carrier is performed as Fluid or Pallet
        self.USPS_Fluid_Status = USPS_Fluid_Status
        self.UPSN_Fluid_Status = UPSN_Fluid_Status
        self.FDEG_Fluid_Status = FDEG_Fluid_Status
        self.FDE_Fluid_Status = FDE_Fluid_Status

        #Used for consideration of breaks
        self.resources_available = True

        #flags used to control whether day shift resources or night shift resources are used.
        self.night_shift = False
        self.day_shift = False

        #flags to control the resources dedicated to processes
        self.inbound_flag = False
        self.TLMD_flag = False
        self.TLMD_sort = False
        self.TLMD_stage = False
        self.LHC_flag = False
        self.TFC_flag = False

        #flags for national carrier progress
        self.USPS_AB_flag = False
        self.UPSN_AB_flag = False
        self.FDEG_AB_flag = False
        self.FDE_ABflag = False
        self.TLMD_AB_flag = False
        self.TLMD_AB_pre_flag = False
        self.inbound_C_flag = False
        self.TLMD_C_flag = False


        self.partition_1_flag = False
        self.partition_2_flag = False
        self.partition_3_flag = False
        self.partition_3AB_flag = False
        self.partition_3C_flag = False

        
        self.partition_1_pallet_flag = False
        self.partition_2_pallet_flag = False
        self.partition_3AB_pallet_flag = False
        self.partition_3C_pallet_flag = False
        
        self.pallet_counter_1 = 1
        self.pallet_counter_2 = 1
        self.pallet_counter_3C = 1
        self.pallet_counter_3AB = 1
        self.TLMD_counter_1 = 0
        self.TLMD_counter_2 = 0
        self.TLMD_counter_3AB = 0
        self.TLMD_counter_3C = 0
        self.pallet_counter_A = 0
        self.process_induct_stage_count = [0]
        self.process_tlmd_stage_count = [0]
        self.process_tlmd_induct_Stage_count = [0]

        self.packages = []

        self.count_removed_tlmd = 0
        self.pickoff_counter = 0
        self.sort_counter = 0
        self.total_packages_1 = 0
        self.total_packages_2 = 0
        self.total_packages_3AB = 0
        self.total_packages_3C = 0
        self.total_packages = 0

        self.received_packages = 0
        self.inducted_packages = 0
        self.sorted_packages = 0


        self.var_status = var_status
        self.var_mag = var_mag
        self.TFC_status = TFC_status

        self.queues = {
            'queue_inbound_truck': simpy.Store(self.env),
            'queue_inbound_staging': simpy.Store(self.env),
            'queue_inbound_staging_packages': simpy.Store(self.env),
            'queue_truck_TFC_packages': simpy.Store(self.env),
            'queue_induct_staging_pallets': simpy.Store(self.env, capacity = 6),
            'queue_induct_staging_packages': simpy.Store(self.env),
            'queue_splitter': simpy.Store(self.env, capacity=1),
            'queue_tlmd_buffer_sort': simpy.Store(self.env, capacity=100),
            'queue_national_carrier_sort': simpy.Store(self.env, capacity=100),
            'queue_tlmd_pallet': simpy.Store(self.env),  
            "queue_FDEG_pallet": simpy.Store(self.env),
            "queue_FDE_pallet": simpy.Store(self.env),
            "queue_USPS_pallet": simpy.Store(self.env),
            "queue_UPSN_pallet": simpy.Store(self.env), 
            'queue_FDEG_staged_pallet': simpy.Store(self.env),
            'queue_FDE_staged_pallet': simpy.Store(self.env),
            'queue_USPS_staged_pallet': simpy.Store(self.env),
            'queue_UPSN_staged_pallet': simpy.Store(self.env),
            "queue_FDEG_fluid": simpy.Store(self.env, capacity=10),
            "queue_FDE_fluid": simpy.Store(self.env, capacity=10),
            "queue_USPS_fluid": simpy.Store(self.env, capacity=10),
            "queue_UPSN_fluid": simpy.Store(self.env, capacity=10),
            'queue_tlmd_partition_1_packages' : simpy.Store(self.env),
            'queue_tlmd_partition_1_packages_f' : simpy.Store(self.env),
            'queue_tlmd_partition_2_packages' : simpy.Store(self.env),
            'queue_tlmd_partition_2_packages_f' : simpy.Store(self.env),
            'queue_tlmd_partition_3AB_packages' : simpy.Store(self.env),
            'queue_tlmd_partition_3AB_packages_f' : simpy.Store(self.env),
            'queue_tlmd_partition_3C_packages' : simpy.Store(self.env),
            'queue_tlmd_partition_3C_packages_f' : simpy.Store(self.env),
            'queue_tlmd_buffer_pallet_1' : simpy.Store(self.env),
            'queue_tlmd_buffer_pallet_2' : simpy.Store(self.env),
            'queue_tlmd_buffer_pallet_3AB' : simpy.Store(self.env),
            'queue_tlmd_buffer_pallet_3C' : simpy.Store(self.env),
            'queue_tlmd_1_staged_pallet': simpy.Store(env),
            'queue_tlmd_2_staged_pallet': simpy.Store(env),
            'queue_tlmd_3_staged_pallet': simpy.Store(env),
            'queue_tlmd_induct_staging_pallets' : simpy.Store(env, capacity=6),
            'queue_tlmd_induct_staging_packages' : simpy.Store(env),
            'queue_tlmd_splitter' : simpy.Store(env, capacity = 1),
            'queue_tlmd_final_sort' : simpy.Store(env, capacity = 100),
            'queue_tlmd_cart' : simpy.Store(env),  
        }

        self.metrics = {
            'processing_times': [],
            'queue_lengths': {key: [] for key in self.queues.keys()},
            'timestamps': [],
            'resource_utilization': [],
        }

        self.all_packages_staged_time = None

    def track_metrics(self):
        while True:
            self.metrics['timestamps'].append(self.env.now)
            for key, queue in self.queues.items():
                self.metrics['queue_lengths'][key].append(len(queue.items))
            yield self.env.timeout(1)



    def schedule_arrivals(self):
        for i, row in self.pallets_df.iterrows():
            pallet = Pallet(
                self.env,
                row['Pallet'],
                row['linehaul'],
                row['packages'],
                row['earliest_arrival']
            )
            self.env.process(self.truck_arrival(pallet))

    def truck_arrival(self, pallet):
        yield self.env.timeout(pallet.pkg_received_utc_ts) 
        pallet.current_queue = 'queue_inbound_truck'
        #print(f'Pallet {pallet.pallet_id} arrived at {self.env.now}')
        #for package in pallet.packages:
            #package.received_ts = self.env.now
        yield self.queues['queue_inbound_truck'].put(pallet)
        self.env.process(self.unload_truck(pallet))

####################################
####inbound induct proces start#####
####################################

    def unload_truck(self, pallet):
        while self.TFC_flag and not self.TLMD_AB_pre_flag:
            yield self.env.timeout(1)
        if self.TFC_flag and self.TLMD_AB_pre_flag:
            if self.TFC_status == True:
                self.env.process(self.fluid_unload_to_packages(pallet))
            else:
                with self.current_resource['tm_TFC_unload'].request(priority=1) as req:
                    yield req
                    yield self.queues['queue_inbound_truck'].get()
                    if self.var_status == True:
                        process_time = np.random.normal(G.UNLOADING_RATE, G.UNLOADING_RATE*self.var_mag)
                    elif self.var_status == False:
                        process_time = G.UNLOADING_RATE
                    yield self.env.timeout(max(0,process_time))  # Unloading time
                    if self.pause_event:
                        while self.pause_event:
                            yield self.env.timeout(1) # Wait for the pause event to be triggered
                    pallet.current_queue = 'queue_inbound_staging'
                    #print(f'Pallet {pallet.pallet_id} unloaded at {self.env.now}')
                    yield self.queues['queue_inbound_staging'].put(pallet)
                    for package in pallet.packages:
                        package.received_ts = self.env.now

                    self.env.process(self.TFC_induct_staging(pallet))
        else:
            with self.current_resource['tm_pit_unload'].request() as req:
                yield req
                yield self.queues['queue_inbound_truck'].get()
                if self.var_status == True:
                    process_time = np.random.normal(G.UNLOADING_RATE, G.UNLOADING_RATE*self.var_mag)
                elif self.var_status == False:
                    process_time = G.UNLOADING_RATE
                yield self.env.timeout(max(0,process_time))  # Unloading time
                if self.pause_event:
                    while self.pause_event:
                        yield self.env.timeout(1) # Wait for the pause event to be triggered
                pallet.current_queue = 'queue_inbound_staging'
                #print(f'Pallet {pallet.pallet_id} unloaded at {self.env.now}')
                yield self.queues['queue_inbound_staging'].put(pallet)
                for package in pallet.packages:
                    package.received_ts = self.env.now

                self.env.process(self.move_to_induct_staging(pallet))

        
    def fluid_unload_to_packages(self, pallet):
        for package in pallet.packages:
            package.current_queue = 'queue_truck_TFC_packages'
            yield self.queues['queue_truck_TFC_packages'].put(package)
            self.env.process(self.fluid_unload(package, pallet))

    def fluid_unload(self, package, pallet):
        while self.TFC_flag and not self.TLMD_AB_pre_flag:
            yield self.env.timeout(1)
        with self.current_resource['tm_TFC_unload'].request(priority=0) as req:
            yield req
            yield self.queues['queue_truck_TFC_packages'].get()
            if self.var_status == True:
                process_time = np.random.normal(G.FLUID_UNLOAD_RATE, G.FLUID_UNLOAD_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.FLUID_UNLOAD_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
            package.received_ts = self.env.now
            package.inducted_ts = self.env.now
            package.current_queue = 'queue_tlmd_buffer_sort'
            yield self.queues['queue_tlmd_buffer_sort'].put(package)
            pallet.current_packages -= 1
            self.env.process(self.TLMD_buffer_TFC_sort(package))

    def TFC_induct_staging(self, pallet):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_TFC_unload'].request(priority=2) as req: 
            yield req
            
            yield self.queues['queue_inbound_staging'].get()
            if self.var_status == True:
                process_time = np.random.normal(G.INDUCT_STAGE_RATE, G.INDUCT_STAGE_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.INDUCT_STAGE_RATE
            yield self.env.timeout(max(0,process_time))  # Unloading time
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
            pallet.current_queue = 'queue_induct_staging_pallets'
            yield self.queues['queue_induct_staging_pallets'].put(pallet)
            #print(f'Pallet {pallet.pallet_id} staged for induction at {self.env.now}')
            for package in pallet.packages:
                package.current_queue = 'queue_induct_staging_packages'
                yield self.queues['queue_induct_staging_packages'].put(package)
                self.env.process(self.TFC_induct_package(package, pallet))

    def TFC_induct_package(self, package, pallet):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_TFC_unload'].request(priority=0) as req:  
            yield req
            yield self.queues['queue_induct_staging_packages'].get()
            if self.var_status == True:
                process_time = np.random.normal(G.INDUCTION_RATE, G.INDUCTION_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.INDUCTION_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
            package.current_queue = 'queue_tlmd_buffer_sort'
            #print(f'Package {package.tracking_number}, {package.scac} inducted at {self.env.now}')
            package.inducted_ts = self.env.now
            yield self.queues['queue_tlmd_buffer_sort'].put(package)
            pallet.current_packages -= 1  # Decrement the counter
            if pallet.current_packages == 0:
                # Remove the pallet from queue_induct_staging_pallets
                self.remove_pallet_from_TFC_queue(pallet)
            self.env.process(self.TLMD_buffer_TFC_sort(package))

    def remove_pallet_from_TFC_queue(self, pallet):
        for i, p in enumerate(self.queues['queue_induct_staging_pallets'].items):
            if p.pallet_id == pallet.pallet_id:
                del self.queues['queue_induct_staging_pallets'].items[i]
                #print(f'Pallet {pallet.pallet_id} removed from queue_induct_staging_pallets at {self.env.now}')
                break

    def move_to_induct_staging(self, pallet):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_pit_induct'].request(priority=1) as req: 
            yield req
            
            yield self.queues['queue_inbound_staging'].get()
            if self.var_status == True:
                process_time = np.random.normal(G.INDUCT_STAGE_RATE, G.INDUCT_STAGE_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.INDUCT_STAGE_RATE
            yield self.env.timeout(max(0,process_time))  # Unloading time
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
            pallet.current_queue = 'queue_induct_staging_pallets'
            yield self.queues['queue_induct_staging_pallets'].put(pallet)
            #print(f'Pallet {pallet.pallet_id} staged for induction at {self.env.now}')
            for package in pallet.packages:
                package.current_queue = 'queue_induct_staging_packages'
                yield self.queues['queue_induct_staging_packages'].put(package)
                self.env.process(self.induct_package(package, pallet))

    def induct_package(self, package, pallet): 

        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_pit_induct'].request(priority=0) as req:  
            yield req
            yield self.queues['queue_induct_staging_packages'].get()
            if self.var_status == True:
                process_time = np.random.normal(G.INDUCTION_RATE, G.INDUCTION_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.INDUCTION_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
            package.current_queue = 'queue_splitter'
            #print(f'Package {package.tracking_number}, {package.scac} inducted at {self.env.now}')
            package.inducted_ts = self.env.now
            yield self.queues['queue_splitter'].put(package)
            pallet.current_packages -= 1  # Decrement the counter
            if pallet.current_packages == 0:
                # Remove the pallet from queue_induct_staging_pallets
                self.remove_pallet_from_queue(pallet)
            self.env.process(self.split_package(package))
        
        
    def remove_pallet_from_queue(self, pallet):
        # Manually search for and remove the pallet from the queue
        for i, p in enumerate(self.queues['queue_induct_staging_pallets'].items):
            if p.pallet_id == pallet.pallet_id:
                del self.queues['queue_induct_staging_pallets'].items[i]
                #print(f'Pallet {pallet.pallet_id} removed from queue_induct_staging_pallets at {self.env.now}')
                break

    def split_package(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_nonpit_split'].request() as req:
            yield req
            yield self.queues['queue_splitter'].get()
            if self.var_status == True:
                process_rate = np.random.normal(G.SPLITTER_RATE, G.SPLITTER_RATE * self.var_mag)
            elif self.var_status == False:
                process_rate = G.SPLITTER_RATE
            yield self.env.timeout(max(0,process_rate))
            if package.scac in ['UPSN', 'USPS', 'FDEG', 'FDE']:
                package.current_queue = 'queue_national_carrier_sort'
                #print(f'Package {package.tracking_number} split to National Sort at {self.env.now}')
                yield self.queues['queue_national_carrier_sort'].put(package)
                self.env.process(self.national_carrier_sort(package))
            else:
                package.current_queue = 'queue_tlmd_buffer_sort'
                #print(f'Package {package.tracking_number} split to TLMD Buffer at {self.env.now}')
                yield self.queues['queue_tlmd_buffer_sort'].put(package)
                self.env.process(self.tlmd_buffer_sort(package))

    def national_carrier_sort(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        with self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_national_carrier_sort"].get()
            if self.var_status == True:
                process_time = np.random.normal(G.NATIONAL_CARRIER_SORT_RATE, G.NATIONAL_CARRIER_SORT_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.NATIONAL_CARRIER_SORT_RATE
            yield self.env.timeout(max(0.,process_time))
            if self.pause_event:
                    while self.pause_event:
                        yield self.env.timeout(1) # Wait for the pause event to be triggered
            if package.scac in ['UPSN']:
                if not self.UPSN_Fluid_Status:
                    yield self.queues["queue_UPSN_pallet"].put(package)
                    package.sorted_ts = self.env.now
                    self.packages.append(package)
                    self.env.process(self.check_all_UPSN_sorted())
                else:
                    yield self.queues["queue_UPSN_fluid"].put(package)
                    package.sorted_ts = self.env.now
                    self.packages.append(package)
                    self.env.process(self.national_carrier_fluid_split_UPSN(package))
            elif package.scac in ['USPS']:
                if not self.USPS_Fluid_Status:
                    yield self.queues["queue_USPS_pallet"].put(package)
                    package.sorted_ts = self.env.now
                    self.packages.append(package)
                    self.env.process(self.check_all_USPS_sorted())
                else:
                    yield self.queues["queue_USPS_fluid"].put(package)
                    package.sorted_ts = self.env.now
                    self.packages.append(package)
                    self.env.process(self.national_carrier_fluid_split_USPS(package))
            elif package.scac in ['FDEG']:
                if not self.FDEG_Fluid_Status:
                    yield self.queues["queue_FDEG_pallet"].put(package)
                    package.sorted_ts = self.env.now
                    self.packages.append(package)
                    self.env.process(self.check_all_FDEG_sorted())
                else:
                    yield self.queues["queue_FDEG_fluid"].put(package)
                    package.sorted_ts = self.env.now
                    self.packages.append(package)
                    self.env.process(self.national_carrier_fluid_split_FDEG(package))
            elif package.scac in ['FDE']:
                if not self.FDE_Fluid_Status:
                    yield self.queues["queue_FDE_pallet"].put(package)
                    package.sorted_ts = self.env.now
                    self.packages.append(package)
                    self.env.process(self.check_all_FDE_sorted())
                else:
                    yield self.queues["queue_FDE_fluid"].put(package)
                    package.sorted_ts = self.env.now
                    self.packages.append(package)
                    self.env.process(self.national_carrier_fluid_split_FDE(package))
        
    def check_all_UPSN_sorted(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        if len(self.queues['queue_UPSN_pallet'].items) == G.UPSN_LINEHAUL_A_PACKAGES + G.UPSN_LINEHAUL_B_PACKAGES:
            #print(f'All A&B UPSN packages sorted at {self.env.now}')
            G.UPSN_SORT_TIME = self.env.now 
            self.UPSN_AB_flag = True
            self.env.process(self.NC_UPSN_pallet_build())
        elif self.UPSN_AB_flag and len(self.queues['queue_UPSN_fluid'].items) == G.UPSN_LINEHAUL_C_PACKAGES:
            #print(f'All C UPSN packages sorted at {self.env.now}')
            self.env.process(self.NC_UPSN_pallet_build())
        else:
            yield self.env.timeout(1)
            self.env.process(self.check_all_UPSN_sorted())

    def check_all_USPS_sorted(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        if len(self.queues['queue_USPS_pallet'].items) == G.USPS_LINEHAUL_A_PACKAGES + G.USPS_LINEHAUL_B_PACKAGES:
            #print(f'All A&B USPS packages sorted at {self.env.now}')
            G.USPS_SORT_TIME = self.env.now
            self.USPS_AB_flag = True
            self.env.process(self.NC_USPS_pallet_build())
        elif self.USPS_AB_flag and len(self.queues['queue_USPS_fluid'].items) == G.USPS_LINEHAUL_C_PACKAGES:
            #print(f'All C USPS packages sorted at {self.env.now}')
            self.env.process(self.NC_USPS_pallet_build())
        else:
            yield self.env.timeout(1)
            self.env.process(self.check_all_USPS_sorted())

    def check_all_FDEG_sorted(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        if  len(self.queues['queue_FDEG_pallet'].items) == G.FDEG_LINEHAUL_A_PACKAGES + G.FDEG_LINEHAUL_B_PACKAGES:
            #print(f'All A&B FDEG packages sorted at {self.env.now}')
            G.FDEG_SORT_TIME = self.env.now
            self.FDEG_AB_flag = True
            self.env.process(self.NC_FDEG_pallet_build())
        elif self.FDEG_AB_flag and len(self.queues['queue_FDEG_fluid'].items) == G.FDEG_LINEHAUL_C_PACKAGES:
            #print(f'All C FDEG packages sorted at {self.env.now}')
            self.env.process(self.NC_FDEG_pallet_build())
        else:
            yield self.env.timeout(1)
            self.env.process(self.check_all_FDEG_sorted())

    def check_all_FDE_sorted(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        if  len(self.queues['queue_FDE_pallet'].items) == G.FDE_LINEHAUL_A_PACKAGES + G.FDE_LINEHAUL_B_PACKAGES:
            #print(f'All A&B FDE packages sorted at {self.env.now}')
            G.FDE_SORT_TIME = self.env.now
            self.FDE__ABflag = True
            self.env.process(self.NC_FDE_pallet_build())
        elif self.FDE_ABflag and len(self.queues['queue_FDE_fluid'].items) == G.FDE_LINEHAUL_C_PACKAGES:
           # print(f'All C FDE packages sorted at {self.env.now}')
            self.env.process(self.NC_FDE_pallet_build())
        else:
            yield self.env.timeout(1)
            self.env.process(self.check_all_FDE_sorted())           

    def NC_UPSN_pallet_build(self):
        UPSN_pallets = math.ceil(G.TOTAL_PACKAGES_UPSN / G.NC_PALLET_MAX_PACKAGES)
        #print(f'UPSN: {UPSN_pallets} pallets')
        G.UPSN_PALLETS = UPSN_pallets

        def create_NC_pallets(NC_packages, NC_pallets, queue_name, staged_queue_name, scac):

            with self.current_resource['tm_nonpit_NC'].request(priority=0) as req:
                yield req
                current_packages = NC_packages
                for pallet_num in range(NC_pallets):
                    pallet_packages = []
                    packages_for_this_pallet = min(G.NC_PALLET_MAX_PACKAGES, current_packages)
                    for _ in range(packages_for_this_pallet):
                        pkg = yield self.queues[queue_name].get()
                        pallet_packages.append(pkg)
                        yield self.env.timeout(0)
                    pallet = National_Carrier_Pallet(self.env, f'Pallet_{G.K}', pallet_packages, scac, self.env.now)
                    G.K += 1
                    #print(f'NC_{scac}, {pallet.pallet_id} created with {len(pallet_packages)} packages at {self.env.now}')
                    yield self.queues[staged_queue_name].put(pallet)
                    if self.var_status == True:
                        process_time = np.random.normal(G.NC_PALLET_STAGING_RATE, G.NC_PALLET_STAGING_RATE * self.var_mag)
                    elif self.var_status == False:
                        process_time = G.NC_PALLET_STAGING_RATE
                    yield self.env.timeout(max(0,process_time))
                    if self.pause_event:
                        while self.pause_event:
                            yield self.env.timeout(1) # Wait for the pause event to be triggered
                    current_packages -= packages_for_this_pallet

        yield self.env.process(create_NC_pallets(G.TOTAL_PACKAGES_UPSN, UPSN_pallets, 'queue_UPSN_pallet', 'queue_UPSN_staged_pallet','UPSN'))

    def NC_USPS_pallet_build(self):
        USPS_pallets = math.ceil(G.TOTAL_PACKAGES_USPS / G.NC_PALLET_MAX_PACKAGES)
        #print(f'USPS: {USPS_pallets} pallets')
        G.USPS_PALLETS = USPS_pallets

        def create_NC_pallets(NC_packages, NC_pallets, queue_name, staged_queue_name, scac):

            with self.current_resource['tm_nonpit_NC'].request(priority=0) as req:
                yield req
                current_packages = NC_packages
                for pallet_num in range(NC_pallets):
                    pallet_packages = []
                    packages_for_this_pallet = min(G.NC_PALLET_MAX_PACKAGES, current_packages)
                    for _ in range(packages_for_this_pallet):
                        pkg = yield self.queues[queue_name].get()
                        pallet_packages.append(pkg)
                        yield self.env.timeout(0)
                    pallet = National_Carrier_Pallet(self.env, f'Pallet_{G.K}', pallet_packages, scac, self.env.now)
                    G.K += 1
                    #print(f'NC_{scac}, {pallet.pallet_id} created with {len(pallet_packages)} packages at {self.env.now}')
                    yield self.queues[staged_queue_name].put(pallet)
                    if self.var_status == True:
                        process_time = np.random.normal(G.NC_PALLET_STAGING_RATE, G.NC_PALLET_STAGING_RATE * self.var_mag)
                    elif self.var_status == False:
                        process_time = G.NC_PALLET_STAGING_RATE
                    yield self.env.timeout(max(0,process_time))
                    if self.pause_event:
                        while self.pause_event:
                            yield self.env.timeout(1) # Wait for the pause event to be triggered
                    current_packages -= packages_for_this_pallet

        yield self.env.process(create_NC_pallets(G.TOTAL_PACKAGES_USPS, USPS_pallets, 'queue_USPS_pallet', 'queue_USPS_staged_pallet','USPS'))


    def NC_FDEG_pallet_build(self):
        FDEG_pallets = math.ceil(G.TOTAL_PACKAGES_FDEG / G.NC_PALLET_MAX_PACKAGES)
        #print(f'FDEG: {FDEG_pallets} pallets')
        G.FDEG_PALLETS = FDEG_pallets

        def create_NC_pallets(NC_packages, NC_pallets, queue_name, staged_queue_name, scac):

            with self.current_resource['tm_nonpit_NC'].request(priority=0) as req:
                yield req
                current_packages = NC_packages
                for pallet_num in range(NC_pallets):
                    pallet_packages = []
                    packages_for_this_pallet = min(G.NC_PALLET_MAX_PACKAGES, current_packages)
                    for _ in range(packages_for_this_pallet):
                        pkg = yield self.queues[queue_name].get()
                        pallet_packages.append(pkg)
                        yield self.env.timeout(0)
                    pallet = National_Carrier_Pallet(self.env, f'Pallet_{G.K}', pallet_packages, scac, self.env.now)
                    G.K += 1
                    #print(f'NC_{scac}, {pallet.pallet_id} created with {len(pallet_packages)} packages at {self.env.now}')
                    yield self.queues[staged_queue_name].put(pallet)
                    if self.var_status == True:
                        process_time = np.random.normal(G.NC_PALLET_STAGING_RATE, G.NC_PALLET_STAGING_RATE * self.var_mag)
                    elif self.var_status == False:
                        process_time = G.NC_PALLET_STAGING_RATE
                    yield self.env.timeout(max(0,process_time))
                    if self.pause_event:
                        while self.pause_event:
                            yield self.env.timeout(1) # Wait for the pause event to be triggered
                    current_packages -= packages_for_this_pallet

        yield self.env.process(create_NC_pallets(G.TOTAL_PACKAGES_FDEG, FDEG_pallets, 'queue_FDEG_pallet', 'queue_FDEG_staged_pallet','FDEG'))


    def NC_FDE_pallet_build(self):
        FDE_pallets = math.ceil(G.TOTAL_PACKAGES_FDE / G.NC_PALLET_MAX_PACKAGES)
        #print(f'FDE: {FDE_pallets} pallets')
        G.FDE_PALLETS = FDE_pallets

        def create_NC_pallets(NC_packages, NC_pallets, queue_name, staged_queue_name, scac):

            with self.current_resource['tm_nonpit_NC'].request(priority=0) as req:
                yield req
                current_packages = NC_packages               
                for pallet_num in range(NC_pallets):
                    pallet_packages = []
                    packages_for_this_pallet = min(G.NC_PALLET_MAX_PACKAGES, current_packages)                       
                    for _ in range(packages_for_this_pallet):
                        pkg = yield self.queues[queue_name].get()
                        pallet_packages.append(pkg)
                        yield self.env.timeout(0)                        
                    pallet = National_Carrier_Pallet(self.env, f'Pallet_{G.K}', pallet_packages, scac, self.env.now)
                    G.K += 1
                    #print(f'NC_{scac}, {pallet.pallet_id} created with {len(pallet_packages)} packages at {self.env.now}')                        
                    yield self.queues[staged_queue_name].put(pallet)
                    if self.var_status == True:
                        process_time = np.random.normal(G.NC_PALLET_STAGING_RATE, G.NC_PALLET_STAGING_RATE * self.var_mag)
                    elif self.var_status == False:
                        process_time = G.NC_PALLET_STAGING_RATE
                    yield self.env.timeout(max(0,process_time))
                    if self.pause_event:
                        while self.pause_event:
                            yield self.env.timeout(1) # Wait for the pause event to be triggered                      
                    current_packages -= packages_for_this_pallet


        yield self.env.process(create_NC_pallets(G.TOTAL_PACKAGES_FDE, FDE_pallets, 'queue_FDE_pallet', 'queue_FDE_staged_pallet','FDE'))

########################################################
##### Begin National Carrier Fluid Load Process#########
########################################################

    def national_carrier_fluid_split_UPSN(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with self.self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_UPSN_fluid"].get()
            if self.var_status == True:
                process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_PICK_RATE, G.NATIONAL_CARRIER_FLUID_PICK_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.NATIONAL_CARRIER_FLUID_PICK_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
            #print(f"Package {package.tracking_number} sorted to UPSN fluid at {self.env.now}")
            yield self.queues["queue_UPSN_truck"].put(package)
            self.env.process(self.national_carrier_fluid_load_UPSN(package))

    def national_carrier_fluid_load_UPSN(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with  self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_UPSN_truck"].get()
            if self.var_status == True:
                process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_LOAD_RATE, G.NATIONAL_CARRIER_FLUID_LOAD_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.NATIONAL_CARRIER_FLUID_LOAD_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
            #print(f"Package {package.tracking_number} fulid loaded to UPSN at {self.env.now}")
            yield self.queues["queue_UPSN_Outbound"].put(package)


    def national_carrier_fluid_split_USPS(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with  self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_USPS_fluid"].get()
            if self.var_status == True:
                process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_PICK_RATE, G.NATIONAL_CARRIER_FLUID_PICK_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.NATIONAL_CARRIER_FLUID_PICK_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
           # print(f"Package {package.tracking_number} sorted to USPS fluid at {self.env.now}")
            yield self.queues["queue_USPS_truck"].put(package)
            self.env.process(self.national_carrier_fluid_load_USPS(package))

    def national_carrier_fluid_load_USPS(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with  self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_USPS_truck"].get()
            if self.var_status == True:
                process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_LOAD_RATE, G.NATIONAL_CARRIER_FLUID_LOAD_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.NATIONAL_CARRIER_FLUID_LOAD_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
           # print(f"Package {package.tracking_number} fulid loaded to USPS at {self.env.now}")
            yield self.queues["queue_USPS_Outbound"].put(package)



    def national_carrier_fluid_split_FDEG(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_FDEG_fluid"].get()
            if self.var_status == True:
                process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_PICK_RATE, G.NATIONAL_CARRIER_FLUID_PICK_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.NATIONAL_CARRIER_FLUID_PICK_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
            #print(f"Package {package.tracking_number} sorted to FDEG fluid at {self.env.now}")
            yield self.queues["queue_FDEG_truck"].put(package)
            self.env.process(self.national_carrier_fluid_load_FDEG(package))


    def national_carrier_fluid_load_FDEG(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_FDEG_truck"].get()
            if self.var_status == True:
                process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_LOAD_RATE, G.NATIONAL_CARRIER_FLUID_LOAD_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.NATIONAL_CARRIER_FLUID_LOAD_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
            #print(f"Package {package.tracking_number} fulid loaded to FDEG at {self.env.now}")
            yield self.queues["queue_FDEG_Outbound"].put(package)


    def national_carrier_fluid_split_FDE(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_FDE_fluid"].get()
            if self.var_status == True:
                process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_PICK_RATE, G.NATIONAL_CARRIER_FLUID_PICK_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.NATIONAL_CARRIER_FLUID_PICK_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
           #print(f"Package {package.tracking_number} sorted to FDE fluid at {self.env.now}")
            yield self.queues["queue_FDE_truck"].put(package)
            self.env.process(self.national_carrier_fluid_load_FDE(package))


    def national_carrier_fluid_load_FDE(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        #while not self.resources_available:
        #    yield self.env.timeout(1)
        with self.current_resource['tm_nonpit_NC'].request(priority=1) as req:
            yield req
            yield self.queues["queue_FDE_truck"].get()
            if self.var_status == True:
                process_time = np.random.normal(G.NATIONAL_CARRIER_FLUID_LOAD_RATE, G.NATIONAL_CARRIER_FLUID_LOAD_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.NATIONAL_CARRIER_FLUID_LOAD_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
           # print(f"Package {package.tracking_number} fulid loaded to FDE at {self.env.now}")
            yield self.queues["queue_FDE_Outbound"].put(package)



    ########################################################
    ############## Begin TLMD Sort Process #################
    ########################################################
    
    def TLMD_buffer_TFC_sort(self, package):
        while self.TFC_flag and not self.TLMD_AB_pre_flag:
            yield self.env.timeout(1)

        with self.current_resource['tm_TFC_sort'].request() as req:
            yield self.queues['queue_tlmd_buffer_sort'].get()
            if self.var_status == True:
                process_time = np.random.normal(G.TLMD_BUFFER_SORT_RATE, G.TLMD_BUFFER_SORT_RATE * self.var_mag)
            else:
                process_time = G.TLMD_BUFFER_SORT_RATE
            yield self.env.timeout(max(0,process_time))
            if self.pause_event:
                while self.pause_event:
                    yield self.env.timeout(1) # Wait for the pause event to be triggered
            #print(f'Package {package.tracking_number} of partition {package.partition} sorted to TLMD Buffer at {self.env.now}')

            if package.partition == '1':
                self.TLMD_counter_1 +=1
                #print(f'partition 1 presorted: {self.TLMD_counter_1}')
                if self.TLMD_counter_1 > ((G.TLMD_PARTITION_1_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_1_packages_f'].put(package)
                    self.env.process(self.check_pallet_1f())
                else:
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_1_packages'].put(package)
                    self.env.process(self.check_pallet_1())
                    
            elif package.partition == '2':
                self.TLMD_counter_2 +=1
                #print(f'partition 2 presorted: {self.TLMD_counter_2}')
                if self.TLMD_counter_2 > ((G.TLMD_PARTITION_2_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_2_packages_f'].put(package)
                    self.env.process(self.check_pallet_2f())
                else:
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_2_packages'].put(package)
                    self.env.process(self.check_pallet_2())
                    
            elif package.partition == '3AB':
                self.TLMD_counter_3AB +=1
                #print(f'partition 3AB presorted: {self.TLMD_counter_3AB}')
                if self.TLMD_counter_3AB > ((G.TLMD_PARTITION_3AB_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_3AB_packages_f'].put(package)
                    self.env.process(self.check_pallet_3ABf())
                else:
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_3AB_packages'].put(package)
                    self.env.process(self.check_pallet_3AB())

            elif package.partition == '3C':
                self.TLMD_counter_3C +=1
                #print(f'partition 3C presorted: {self.TLMD_counter_3C}')
                if self.TLMD_counter_3C > ((G.TLMD_PARTITION_3C_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_3C_packages_f'].put(package)
                    self.env.process(self.check_pallet_3Cf())
                else:
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_3C_packages'].put(package)
                    self.env.process(self.check_pallet_3C())

        

 ######################################################


      ########################################################
    ############## Begin TLMD Sort Process #################
    ########################################################
    
    # this will need to be updated to include the logic associated with the different partitions
    def tlmd_buffer_sort(self, package):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        if self.pause_event:
            while self.pause_event:
                yield self.env.timeout(1) # Wait for the pause event to be triggered
        with self.current_resource['tm_nonpit_buffer'].request(priority=1) as req:
            yield req
            yield self.queues['queue_tlmd_buffer_sort'].get()
            if self.var_status == True:
                process_time = np.random.normal(G.TLMD_BUFFER_SORT_RATE, G.TLMD_BUFFER_SORT_RATE * self.var_mag)
            else:
                process_time = G.TLMD_BUFFER_SORT_RATE
            yield self.env.timeout(max(0,process_time))
            #print(f'Package {package.tracking_number} of partition {package.partition} sorted to TLMD Buffer at {self.env.now}')

            if package.partition == '1':
                self.TLMD_counter_1 +=1
                #print(f'partition 1 presorted: {self.TLMD_counter_1}')
                if self.TLMD_counter_1 > ((G.TLMD_PARTITION_1_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_1_packages_f'].put(package)
                    self.env.process(self.check_pallet_1f())
                else:
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_1_packages'].put(package)
                    self.env.process(self.check_pallet_1())
                    
            elif package.partition == '2':
                self.TLMD_counter_2 +=1
                #print(f'partition 2 presorted: {self.TLMD_counter_2}')
                if self.TLMD_counter_2 > ((G.TLMD_PARTITION_2_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_2_packages_f'].put(package)
                    self.env.process(self.check_pallet_2f())
                else:
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_2_packages'].put(package)
                    self.env.process(self.check_pallet_2())
                    
            elif package.partition == '3AB':
                self.TLMD_counter_3AB +=1
                #print(f'partition 3AB presorted: {self.TLMD_counter_3AB}')
                if self.TLMD_counter_3AB > ((G.TLMD_PARTITION_3AB_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_3AB_packages_f'].put(package)
                    self.env.process(self.check_pallet_3ABf())
                else:
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_3AB_packages'].put(package)
                    self.env.process(self.check_pallet_3AB())
            elif package.partition == '3C':
                self.TLMD_counter_3C +=1
                #print(f'partition 3C presorted: {self.TLMD_counter_3C}')
                if self.TLMD_counter_3C > ((G.TLMD_PARTITION_3C_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_3C_packages_f'].put(package)
                    self.env.process(self.check_pallet_3Cf())
                else:
                    delay = random.uniform(0, 0.05)
                    yield self.env.timeout(delay)
                    self.queues['queue_tlmd_partition_3C_packages'].put(package)
                    self.env.process(self.check_pallet_3C())
        if self.TLMD_counter_1 + self.TLMD_counter_2 + self.TLMD_counter_3AB == G.TLMD_LINEHAUL_A_PACKAGES + G.TLMD_LINEHAUL_B_PACKAGES:
                self.TLMD_AB_pre_flag = True
        if self.TLMD_counter_3C == G.TLMD_LINEHAUL_C_PACKAGES:
                self.TLMD_C_pre_flag = True

 ######################################################


    def check_pallet_1(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        if len(self.queues['queue_tlmd_partition_1_packages'].items) < G.TLMD_PARTITION_PALLET_MAX_PACKAGES:
            yield self.env.timeout(1)
            self.env.process(self.check_pallet_1())
        else:
            pallet_packages_1 = []
            for _ in range(G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                pkg = yield self.queues['queue_tlmd_partition_1_packages'].get()
                pallet_packages_1.append(pkg)
                yield self.env.timeout(0)
            pallet = TLMD_Pallet(self.env, f'TLMD_pallet_1_{G.I}', pallet_packages_1, self.env.now)
            self.pallet_counter_1 += 1
            G.I += 1
            #print(f'TLMD {pallet.pallet_id} created with {len(pallet_packages_1)} packages at {self.env.now}')
            yield self.env.timeout(1)
            yield self.queues['queue_tlmd_buffer_pallet_1'].put(pallet)
            self.env.process(self.stage_pallets())

    def check_pallet_1f(self):
        while self.LHC_flag and not self.partition_1_flag:
            yield self.env.timeout(1)
        packages_for_this_pallet_1f = G.TLMD_PARTITION_1_PACKAGES - ((G.TLMD_PARTITION_1_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
        if len(self.queues['queue_tlmd_partition_1_packages_f'].items) < packages_for_this_pallet_1f:
            yield self.env.timeout(1)
            self.env.process(self.check_pallet_1f())
        else:
            pallet_packages_1f = []
            for _ in range(packages_for_this_pallet_1f):
                pkg = yield self.queues['queue_tlmd_partition_1_packages_f'].get()
                pallet_packages_1f.append(pkg)
                yield self.env.timeout(0)
            pallet = TLMD_Pallet(self.env, f'TLMD_pallet_1_{G.I}', pallet_packages_1f, self.env.now)
            self.pallet_counter_1 += 1
            if not self.partition_1_pallet_flag:
                self.partition_1_pallet_flag = True
            G.I += 1
            #print(f'Final partition 1 pallet made at {self.env.now}')
            #print(f'Partition 1 packages buffered: {self.TLMD_counter_1}')
            #print(f'total 1 pallets generated: {self.pallet_counter_1 -1}')
            #print(f'TLMD {pallet.pallet_id} created with {len(pallet_packages_1f)} packages at {self.env.now}')

            yield self.env.timeout(1)
            yield self.queues['queue_tlmd_buffer_pallet_1'].put(pallet)
            self.env.process(self.stage_pallets())

    def check_pallet_2(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        if len(self.queues['queue_tlmd_partition_2_packages'].items) < G.TLMD_PARTITION_PALLET_MAX_PACKAGES:
            yield self.env.timeout(1)
            self.env.process(self.check_pallet_2())
        else:
            pallet_packages_2 = []
            for _ in range(G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                pkg = yield self.queues['queue_tlmd_partition_2_packages'].get()
                pallet_packages_2.append(pkg)
                yield self.env.timeout(0)
            pallet = TLMD_Pallet(self.env, f'TLMD_pallet_2_{G.L}', pallet_packages_2, self.env.now)
            self.pallet_counter_2 += 1
            G.L += 1
            #print(f'TLMD {pallet.pallet_id} created with {len(pallet_packages_2)} packages at {self.env.now}')
            yield self.env.timeout(1)
            yield self.queues['queue_tlmd_buffer_pallet_2'].put(pallet)
            self.env.process(self.stage_pallets())

    def check_pallet_2f(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        packages_for_this_pallet_2f = G.TLMD_PARTITION_2_PACKAGES - ((G.TLMD_PARTITION_2_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
        if len(self.queues['queue_tlmd_partition_2_packages_f'].items) < packages_for_this_pallet_2f:
            yield self.env.timeout(1)
            self.env.process(self.check_pallet_2f())
        else:
            pallet_packages_2f = []
            for _ in range(packages_for_this_pallet_2f):
                pkg = yield self.queues['queue_tlmd_partition_2_packages_f'].get()
                pallet_packages_2f.append(pkg)
                yield self.env.timeout(0)
            pallet = TLMD_Pallet(self.env, f'TLMD_pallet_2_{G.L}', pallet_packages_2f, self.env.now)
            self.pallet_counter_2 += 1
            if not self.partition_2_pallet_flag:
                self.partition_2_pallet_flag = True
            G.L += 1
            #print(f'Final partition 2 pallet made at {self.env.now}')
            #print(f'Partition 2 packages buffered: {self.TLMD_counter_2}')
            #print(f'total 2 pallets generated: {self.pallet_counter_2 -1}')
            #print(f'TLMD {pallet.pallet_id} created with {len(pallet_packages_2f)} packages at {self.env.now}')
            yield self.env.timeout(1)
            yield self.queues['queue_tlmd_buffer_pallet_2'].put(pallet)
            self.env.process(self.stage_pallets())
            

    ###########################################################
    def check_pallet_3AB(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        if len(self.queues['queue_tlmd_partition_3AB_packages'].items) < G.TLMD_PARTITION_PALLET_MAX_PACKAGES:
            yield self.env.timeout(1)
            self.env.process(self.check_pallet_3AB())
        else:
            pallet_packages_3AB = []
            for _ in range(G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                pkg = yield self.queues['queue_tlmd_partition_3AB_packages'].get()
                pallet_packages_3AB.append(pkg)
                yield self.env.timeout(0)
            pallet = TLMD_Pallet(self.env, f'TLMD_pallet_3_{G.M}', pallet_packages_3AB, self.env.now)
            self.pallet_counter_3AB += 1
            G.M += 1
            #print(f'TLMD {pallet.pallet_id} created with {len(pallet_packages_3AB)} packages at {self.env.now}')
            yield self.env.timeout(1)
            yield self.queues['queue_tlmd_buffer_pallet_3AB'].put(pallet)
            self.env.process(self.stage_pallets())

    def check_pallet_3ABf(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        packages_for_this_pallet_3ABf = G.TLMD_PARTITION_3AB_PACKAGES - ((G.TLMD_PARTITION_3AB_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES)

        if len(self.queues['queue_tlmd_partition_3AB_packages_f'].items) < packages_for_this_pallet_3ABf:
            yield self.env.timeout(1)
            self.env.process(self.check_pallet_3ABf())
        else:
            
            pallet_packages_3ABf = []
            for _ in range(packages_for_this_pallet_3ABf):
                pkg = yield self.queues['queue_tlmd_partition_3AB_packages_f'].get()
                pallet_packages_3ABf.append(pkg)
                yield self.env.timeout(0)
            pallet = TLMD_Pallet(self.env, f'TLMD_pallet_3_{G.M}', pallet_packages_3ABf, self.env.now)
            self.pallet_counter_3AB += 1
            if not self.partition_3AB_pallet_flag:
                self.partition_3AB_pallet_flag = True
            G.M += 1
            #print(f'Final partition 3AB pallet made at {self.env.now}')
            #print(f'Partition 3AB packages buffered: {self.TLMD_counter_3AB}')
            #print(f'total 3AB pallets generated: {self.pallet_counter_3AB -1}')
            #print(f'TLMD {pallet.pallet_id} created with {len(pallet_packages_3ABf)} packages at {self.env.now}')
            yield self.env.timeout(1)
            yield self.queues['queue_tlmd_buffer_pallet_3AB'].put(pallet)
            self.env.process(self.stage_pallets())

###############################################################
    def check_pallet_3C(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)

        if len(self.queues['queue_tlmd_partition_3C_packages'].items) < G.TLMD_PARTITION_PALLET_MAX_PACKAGES:
            yield self.env.timeout(1)
            self.env.process(self.check_pallet_3C())
        else:
            pallet_packages_3C = []
            for _ in range(G.TLMD_PARTITION_PALLET_MAX_PACKAGES):
                pkg = yield self.queues['queue_tlmd_partition_3C_packages'].get()
                pallet_packages_3C.append(pkg)
                yield self.env.timeout(0)
            pallet = TLMD_Pallet(self.env, f'TLMD_pallet_3_{G.M}', pallet_packages_3C, self.env.now)
            self.pallet_counter_3C += 1
            G.M += 1
            #print(f'TLMD {pallet.pallet_id} created with {len(pallet_packages_3C)} packages at {self.env.now}')
            yield self.queues['queue_tlmd_buffer_pallet_3C'].put(pallet)
            self.env.process(self.stage_pallets())

    def check_pallet_3Cf(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        
        packages_for_this_pallet_3Cf = G.TLMD_PARTITION_3C_PACKAGES - ((G.TLMD_PARTITION_3C_PALLETS - 1) * G.TLMD_PARTITION_PALLET_MAX_PACKAGES)

        if len(self.queues['queue_tlmd_partition_3C_packages_f'].items) < packages_for_this_pallet_3Cf:
            yield self.env.timeout(1)
            self.env.process(self.check_pallet_3Cf())
        else:
            if not self.partition_3C_pallet_flag:
                pallet_packages_3Cf = []
                for _ in range(packages_for_this_pallet_3Cf):
                    pkg = yield self.queues['queue_tlmd_partition_3C_packages_f'].get()
                    pallet_packages_3Cf.append(pkg)
                    yield self.env.timeout(0)
                pallet = TLMD_Pallet(self.env, f'TLMD_pallet_3_{G.M}', pallet_packages_3Cf, self.env.now)
                self.pallet_counter_3C += 1
                G.M += 1
                #print(f'TLMD {pallet.pallet_id} created with {len(pallet_packages_3Cf)} packages at {self.env.now}')
                #print(f'Final partition 3C pallet made at {self.env.now}')
                #print(f'total 3C pallets generated: {self.pallet_counter_3C -1}')
                yield self.queues['queue_tlmd_buffer_pallet_3C'].put(pallet)
                self.partition_3C_pallet_flag = True
                self.env.process(self.stage_pallets())
    
    def stage_pallets(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        if self.TLMD_counter_1 + self.TLMD_counter_2 + self.TLMD_counter_3AB >= G.TLMD_LINEHAUL_A_PACKAGES + G.TLMD_LINEHAUL_B_PACKAGES:
                self.TLMD_AB_pre_flag = True
        if self.pause_event:
            while self.pause_event:
                yield self.env.timeout(1) # Wait for the pause event to be triggered
        with self.current_resource['tm_nonpit_buffer'].request(priority=1) as req:
            yield req
            if len(self.queues['queue_tlmd_buffer_pallet_1'].items) > 0:
                pallet = yield self.queues['queue_tlmd_buffer_pallet_1'].get()
                if self.var_status == True:
                    process_time = np.random.normal(G.TLMD_PARTITION_STAGE_RATE, G.TLMD_PARTITION_STAGE_RATE * self.var_mag)
                elif self.var_status == False:
                    process_time = G.TLMD_PARTITION_STAGE_RATE
                yield self.env.timeout(max(0,process_time))

                pallet.current_queue = 'queue_tlmd_1_staged_pallet'
                yield self.queues['queue_tlmd_1_staged_pallet'].put(pallet)

            elif len(self.queues['queue_tlmd_buffer_pallet_2'].items) > 0:
                pallet = yield self.queues['queue_tlmd_buffer_pallet_2'].get()
                if self.var_status == True:
                    process_time = np.random.normal(G.TLMD_PARTITION_STAGE_RATE, G.TLMD_PARTITION_STAGE_RATE * self.var_mag)
                elif self.var_status == False:
                    process_time = G.TLMD_PARTITION_STAGE_RATE
                yield self.env.timeout(max(0,process_time))
                pallet.current_queue = 'queue_tlmd_2_staged_pallet'
                yield self.queues['queue_tlmd_2_staged_pallet'].put(pallet)

            elif len(self.queues['queue_tlmd_buffer_pallet_3AB'].items) > 0:
                pallet = yield self.queues['queue_tlmd_buffer_pallet_3AB'].get()
                if self.var_status == True:
                    process_time = np.random.normal(G.TLMD_PARTITION_STAGE_RATE, G.TLMD_PARTITION_STAGE_RATE * self.var_mag)
                elif self.var_status == False:
                    process_time = G.TLMD_PARTITION_STAGE_RATE
                yield self.env.timeout(max(0,process_time))
                pallet.current_queue = 'queue_tlmd_3_staged_pallet'
                yield self.queues['queue_tlmd_3_staged_pallet'].put(pallet)

            elif len(self.queues['queue_tlmd_buffer_pallet_3C'].items) > 0:
                pallet = yield self.queues['queue_tlmd_buffer_pallet_3C'].get()
                if self.var_status == True:
                    process_time = np.random.normal(G.TLMD_PARTITION_STAGE_RATE, G.TLMD_PARTITION_STAGE_RATE * self.var_mag)
                elif self.var_status == False:
                    process_time = G.TLMD_PARTITION_STAGE_RATE
                yield self.env.timeout(max(0,process_time))
                pallet.current_queue = 'queue_tlmd_3_staged_pallet'
                yield self.queues['queue_tlmd_3_staged_pallet'].put(pallet)
            else:
                print('ERROR: No pallets left in any queue')
                return
            self.pallet_counter_A += 1
            #print(f'Pallet {pallet.pallet_id} staged at time {self.env.now}')
            self.env.process(self.check_inbound_finished())

    def check_inbound_finished(self):
        while self.LHC_flag and not self.partition_2_flag:
            yield self.env.timeout(1)
        while not self.partition_1_pallet_flag or not self.partition_2_pallet_flag or not self.partition_3AB_pallet_flag and not self.TLMD_AB_flag and not self.LHC_flag:
            yield self.env.timeout(1)

        self.total_packages_1 = 0
        self.total_packages_2 = 0
        self.total_packages_3AB = 0
        
        #for pallet in self.queues['queue_tlmd_1_staged_pallet'].items:
           # self.total_packages_1 += len(pallet.packages)
        #print(f'Total packages in buffer on 1 pallets: {self.total_packages_1}')
        #for pallet in self.queues['queue_tlmd_2_staged_pallet'].items:
            #self.total_packages_2 += len(pallet.packages)
        #print(f'Total packages in buffer on 2 pallets: {self.total_packages_2}')
        #for pallet in self.queues['queue_tlmd_3_staged_pallet'].items:
            #self.total_packages_3AB += len(pallet.packages)
        #print(f'Total packages in buffer on 3AB pallets: {self.total_packages_3AB}')
        if self.partition_1_pallet_flag and self.partition_2_pallet_flag and self.partition_3AB_pallet_flag and not self.TLMD_AB_flag:
            if self.pallet_counter_1 != G.TLMD_PARTITION_1_PALLETS or self.pallet_counter_2 != G.TLMD_PARTITION_2_PALLETS or self.pallet_counter_3AB != G.TLMD_PARTITION_3AB_PALLETS and not self.TLMD_AB_flag and not self.LHC_flag:
                G.PASSED_OVER_PACKAGES = "ERROR"
        if len(self.queues["queue_tlmd_1_staged_pallet"].items) == G.TLMD_PARTITION_1_PALLETS and len(self.queues["queue_tlmd_2_staged_pallet"].items) == G.TLMD_PARTITION_2_PALLETS and len(self.queues["queue_tlmd_3_staged_pallet"].items) == G.TLMD_PARTITION_3AB_PALLETS and not self.TLMD_AB_flag and not self.LHC_flag:
            self.ab_TLMD_packages_staged_time = self.env.now
            G.TLMD_AB_INDUCT_TIME = self.ab_TLMD_packages_staged_time
            #print(f'All tlmd A&B packages staged at {self.ab_TLMD_packages_staged_time}')
            #print(f' total packages processed: {self.TLMD_counter_1 + self.TLMD_counter_2 + self.TLMD_counter_3AB}')
            #print(f'Total linehaul AB packages: {G.TLMD_LINEHAUL_A_PACKAGES + G.TLMD_LINEHAUL_B_PACKAGES + G.TLMD_LINEHAUL_TFC_PACKAGES}')
            #print(f'Partition 1 pallets: {len(self.queues["queue_tlmd_1_staged_pallet"].items)}')
            #print(f'Partition 2 pallets: {len(self.queues["queue_tlmd_2_staged_pallet"].items)}')
            #print(f'Partition 3AB pallets: {len(self.queues["queue_tlmd_3_staged_pallet"].items)}')
            self.TLMD_AB_flag = True
            self.TFC_flag = False
            G.TLMD_STAGED_PACKAGES = len(self.queues['queue_tlmd_1_staged_pallet'].items) + len(self.queues['queue_tlmd_2_staged_pallet'].items) + len(self.queues['queue_tlmd_3_staged_pallet'].items)
        
        while not self.partition_3C_pallet_flag and self.LHC_flag:
            yield self.env.timeout(1)
        if self.partition_3C_pallet_flag and self.LHC_flag:
            self.c_TLDM_packages_staged_time = self.env.now
            G.TLMD_C_INDUCT_TIME = self.c_TLDM_packages_staged_time
            #print(f'C pallets are staged at {self.c_TLDM_packages_staged_time}')
            #print(f' total packages processed: {self.TLMD_counter_1 + self.TLMD_counter_2 + self.TLMD_counter_3AB + self.TLMD_counter_3C}')
            #print(f'Total linehaul ABC packages: {G.TLMD_LINEHAUL_A_PACKAGES + G.TLMD_LINEHAUL_B_PACKAGES + G.TLMD_LINEHAUL_TFC_PACKAGES + G.TLMD_LINEHAUL_C_PACKAGES}')
            #print(f'Total Pallets staged: {self.pallet_counter_A}')
            self.TLMD_C_flag = True
            self.LHC_flag = False
        self.env.process(self.feed_TLMD_induct_staging())


    def feed_TLMD_induct_staging(self):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)
        while not self.TLMD_AB_flag:
            yield self.env.timeout(1)
        #print(f'TLMD A&B induction started at {self.env.now}')
        if self.pause_event:
            while self.pause_event:
                yield self.env.timeout(1) # Wait for the pause event to be triggered
        with self.current_resource['tm_TLMD_induct_stage'].request(priority=0) as req:
            yield req
            #if  self.process_tlmd_induct_Stage_count[0] < 1:
                #self.process_tlmd_induct_Stage_count[0] += 1
            while len(self.queues['queue_tlmd_induct_staging_pallets'].items) >= (self.queues['queue_tlmd_induct_staging_pallets'].capacity -2):
                yield self.env.timeout(1)
            if len(self.queues['queue_tlmd_1_staged_pallet'].items) > 0:
                pallet = yield self.queues['queue_tlmd_1_staged_pallet'].get()
            elif len(self.queues['queue_tlmd_2_staged_pallet'].items) > 0:
                pallet = yield self.queues['queue_tlmd_2_staged_pallet'].get()
            elif len(self.queues['queue_tlmd_3_staged_pallet'].items) > 0:
                pallet = yield self.queues['queue_tlmd_3_staged_pallet'].get()
            else:
                print('ERROR: No pallets left in any queue')
                return
            if self.var_status == True:
                process_time = np.random.normal(G.TLMD_INDUCT_STAGE_RATE, G.TLMD_INDUCT_STAGE_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.TLMD_INDUCT_STAGE_RATE
            delay = random.uniform(0, 0.005)
            yield self.env.timeout(max(0,process_time+delay))
            
            pallet.current_queue = 'queue_tlmd_induct_staging_pallets'
            yield self.queues['queue_tlmd_induct_staging_pallets'].put(pallet)
            #self.process_tlmd_induct_Stage_count[0] -=1
            #print(f'Pallet {pallet.pallet_id} staged for TLMD induction at {self.env.now}')
            #print(f'Pallets remaining from partition 1: {len(self.queues["queue_tlmd_1_staged_pallet"].items)}')
            #print(f'Pallets remaining from partition 2: {len(self.queues["queue_tlmd_2_staged_pallet"].items)}')
            #print(f'Pallets remaining from partition 3AB: {len(self.queues["queue_tlmd_3_staged_pallet"].items)}')
            for package in pallet.packages:
                package.current_queue = 'queue_tlmd_induct_staging_packages'
                yield self.queues['queue_tlmd_induct_staging_packages'].put(package)
                self.env.process(self.tlmd_induct_package(package, pallet))
    

    def tlmd_induct_package(self, package, pallet):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)
        if self.pause_event:
            while self.pause_event:
                yield self.env.timeout(1) # Wait for the pause event to be triggered

        with self.current_resource['tm_TLMD_induct'].request(priority=0) as req:  
            yield req
            yield self.queues['queue_tlmd_induct_staging_packages'].get()
            if self.var_status == True:
                process_time = np.random.normal(G.TLMD_INDUCTION_RATE, G.TLMD_INDUCTION_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.TLMD_INDUCTION_RATE
            yield self.env.timeout(max(0,process_time))

            package.current_queue = 'queue_tlmd_splitter'
            #print(f'Package {package.tracking_number}, {package.scac} inducted at TLMD at {self.env.now}')
            yield self.queues['queue_tlmd_splitter'].put(package)
            pallet.current_packages -= 1  
            if pallet.current_packages == 0:
                # Remove the pallet from queue_induct_staging_pallets
                #print(f'Pallet {pallet.pallet_id} removed from queue {self.env.now}')
                self.tlmd_remove_pallet_from_queue(pallet)
            self.env.process(self.tlmd_lane_pickoff(package))

    def tlmd_remove_pallet_from_queue(self, pallet):
        # Manually search for and remove the pallet from the queue
        for i, p in enumerate(self.queues['queue_tlmd_induct_staging_pallets'].items):
            if p.pallet_id == pallet.pallet_id:
                del self.queues['queue_tlmd_induct_staging_pallets'].items[i]
                #print(f'Pallet {pallet.pallet_id} removed from queue_tlmd_induct_staging_pallets at {self.env.now}')
                self.count_removed_tlmd += 1
                #print(f'Total pallets removed: {self.count_removed_tlmd} at time: {self.env.now}')
                break


    def tlmd_lane_pickoff(self, package):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)
        if self.pause_event:
            while self.pause_event:
                yield self.env.timeout(1)
        with self.current_resource['tm_TLMD_picker'].request() as req:
            yield req
            yield self.queues['queue_tlmd_splitter'].get()
            if self.var_status == True:
                process_time = np.random.normal(G.SPLITTER_RATE, G.SPLITTER_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.SPLITTER_RATE
            yield self.env.timeout(max(0,process_time))
             # Wait for the pause event to be triggered
            package.current_queue = 'queue_tlmd_final_sort'
            #print(f'Package {package.tracking_number} split to TLMD Final Sort at {self.env.now}')
            yield self.queues['queue_tlmd_final_sort'].put(package)
            self.pickoff_counter += 1
            #print(f'Packages at pickoff: {self.pickoff_counter} at time {self.env.now}')
            self.env.process(self.tlmd_final_sort(package))
    
    def tlmd_final_sort(self, package):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)
        
        if self.pause_event:
            while self.pause_event:
                yield self.env.timeout(1)
        
        with self.current_resource['tm_TLMD_sort'].request() as req:
            yield req
            yield self.queues['queue_tlmd_final_sort'].get()
            if self.var_status == True:
                process_time = np.random.normal(G.TLMD_FINAL_SORT_RATE, G.TLMD_FINAL_SORT_RATE * self.var_mag)
            elif self.var_status == False:
                process_time = G.TLMD_FINAL_SORT_RATE
            #print(f'Final Sort Process Rate: {process_time}')
            yield self.env.timeout(max(0,process_time))
            package.sorted_ts = self.env.now
            self.packages.append(package)
             # Wait for the pause event to be triggered
            #print(f'Package {package.tracking_number} sorted to TLMD Cart at {self.env.now}')
            yield self.queues['queue_tlmd_cart'].put(package)
            self.sort_counter += 1
            #print(f'Packages at final sort: {self.sort_counter} at time {self.env.now}')
            
            if self.sort_counter >= G.TLMD_PARTITION_1_PACKAGES and not self.partition_1_flag:
                G.TLMD_PARTITION_1_SORT_TIME = self.env.now
                self.partition_1_flag = True
                #print(f'Partition 1 done: {self.env.now} with {self.sort_counter} packages')
            if self.sort_counter >= G.TLMD_PARTITION_1_PACKAGES + G.TLMD_PARTITION_2_PACKAGES and not self.partition_2_flag:
                G.TLMD_PARTITION_2_SORT_TIME = self.env.now
                self.partition_2_flag = True
                #print(f'Partition 2 done: {self.env.now} with {self.sort_counter} packages')
            if self.sort_counter >= G.TLMD_PARTITION_1_PACKAGES + G.TLMD_PARTITION_2_PACKAGES + G.TLMD_PARTITION_3AB_PACKAGES and not self.partition_3AB_flag:
                G.TLMD_PARTITION_3AB_SORT_TIME = self.env.now
                self.partition_3AB_flag = True
                #print(f'Partition 3AB done: {self.env.now} with {self.sort_counter} packages')
            if self.sort_counter >= G.TLMD_PARTITION_1_PACKAGES + G.TLMD_PARTITION_2_PACKAGES + G.TLMD_PARTITION_3AB_PACKAGES + G.TLMD_PARTITION_3C_PACKAGES and not self.partition_3C_flag:
                G.TLMD_PARTITION_3_SORT_TIME = self.env.now
                self.partition_3C_flag = True
                G.TLMD_SORTED_PACKAGES = len(self.queues['queue_tlmd_cart'].items)
                #print(f'Partition 3C done: {self.env.now} with {self.sort_counter} packages')
            self.env.process(self.check_all_TLMD_sorted())


    def check_all_TLMD_sorted(self):
        while self.LHC_flag and self.partition_2_flag:
            yield self.env.timeout(1)

        if self.pause_event:
            while self.pause_event:
                yield self.env.timeout(1)
        if self.partition_1_flag and self.partition_2_flag and self.partition_3AB_flag and self.partition_3AB_flag: 
            G.PASSED_OVER_PACKAGES = 0
            if G.TOTAL_PACKAGES_TLMD - len(self.queues['queue_tlmd_cart'].items) == 0:
                #print(f'All Packages sorted at {self.env.now}')
                yield self.env.timeout(1)
        
        else:
            G.PASSED_OVER_PACKAGES = G.TOTAL_PACKAGES_TLMD - len(self.queues['queue_tlmd_cart'].items)

        yield self.env.timeout(1)
        #yield self.env.process(self.check_all_TLMD_sorted())

    #def TLMD_cart_build(self):
       # while self.LHC_flag and self.partition_2_flag:
            
        

    def run_simulation(self, until):
        return self.env.timeout(until)

    
#####################################################################################

def setup_simulation(day_pallets, 
                    night_tm_pit_unload, 
                    night_tm_pit_induct, 
                    night_tm_nonpit_split, 
                    night_tm_nonpit_NC, 
                    night_tm_nonpit_buffer,
                    night_tm_TLMD_induct,
                    night_tm_TLMD_induct_stage,
                    night_tm_TLMD_picker,
                    night_tm_TLMD_sort,
                    night_tm_TLMD_stage,
                    day_tm_pit_unload,
                    day_tm_pit_induct,
                    day_tm_nonpit_split,
                    day_tm_nonpit_NC,
                    day_tm_nonpit_buffer,
                    day_tm_TLMD_induct,
                    day_tm_TLMD_induct_stage,
                    day_tm_TLMD_picker,
                    day_tm_TLMD_sort,
                    day_tm_TLMD_stage,
                    USPS_Fluid_Status,
                    UPSN_Fluid_Status,
                    FDEG_Fluid_Status,
                    FDE_Fluid_Status,
                    var_status,
                    process_variance,
                    TFC_status
                    ):
        

    def sum_packages_by_linehaul(df, linehaul):
        filtered_df = df[df['linehaul'] == linehaul]
        all_lh_packages = sum([len(row['packages']) for i, row in filtered_df.iterrows()])
        return all_lh_packages
    
    total_packages = sum([len(row['packages']) for i, row in day_pallets.iterrows()])
    all_packages = [pkg for sublist in day_pallets['packages'] for pkg in sublist]
    total_packages_NC = len([pkg for pkg in all_packages if pkg[1] in ['UPSN', 'USPS', 'FDEG', 'FDE']])
    total_packages_UPSN = len([pkg for pkg in all_packages if pkg[1] in ['UPSN']])
    total_packages_USPS = len([pkg for pkg in all_packages if pkg[1] in ['USPS']])
    total_packages_FDEG = len([pkg for pkg in all_packages if pkg[1] in ['FDEG']])
    total_packages_FDE = len([pkg for pkg in all_packages if pkg[1] in ['FDE']])
    linehaul_A_packages = sum_packages_by_linehaul(day_pallets, 'A')
    linehaul_B_packages = sum_packages_by_linehaul(day_pallets, 'B')
    linehaul_C_packages = sum_packages_by_linehaul(day_pallets, 'C')
    linehaul_TFC_packages = sum_packages_by_linehaul(day_pallets, 'TFC')

    def filter_packages_by_linehaul(df, linehaul):
        return df[df['linehaul'] == linehaul]

    def sum_packages_by_type(filtered_df, package_type):
        all_packages = [pkg for sublist in filtered_df['packages'] for pkg in sublist]
        target_packages = len([pkg for pkg in all_packages if pkg[1] in [package_type]])
        return target_packages


    # Filter DataFrames by linehaul
    linehaul_A_df = filter_packages_by_linehaul(day_pallets, 'A')
    linehaul_B_df = filter_packages_by_linehaul(day_pallets, 'B')
    linehaul_C_df = filter_packages_by_linehaul(day_pallets, 'C')
    linehaul_TFC_df = filter_packages_by_linehaul(day_pallets, 'TFC')

    total_packages_TLMD = total_packages - total_packages_NC

    G.LINEHAUL_C_TIME = linehaul_C_df['earliest_arrival'].min()
    G.LINEHAUL_TFC_TIME = linehaul_TFC_df['earliest_arrival'].iloc[0]

    G.TOTAL_PACKAGES = total_packages
    G.TOTAL_PACKAGES_TLMD = total_packages_TLMD
    G.TOTAL_PACKAGES_NC = total_packages_NC 
    G.TOTAL_PACKAGES_UPSN = total_packages_UPSN
    G.TOTAL_PACKAGES_USPS = total_packages_USPS
    G.TOTAL_PACKAGES_FDEG = total_packages_FDEG
    G.TOTAL_PACKAGES_FDE = total_packages_FDE
    G.TOTAL_LINEHAUL_A_PACKAGES = linehaul_A_packages
    G.TOTAL_LINEHAUL_B_PACKAGES = linehaul_B_packages
    G.TOTAL_LINEHAUL_C_PACKAGES = linehaul_C_packages

    if G.TOTAL_PACKAGES_UPSN == 0:
        G.UPSN_SORT_TIME = 0
        G.UPSN_PALLETS = 0
    if G.TOTAL_PACKAGES_USPS == 0:
        G.USPS_SORT_TIME = 0
        G.USPS_PALLETS = 0
    if G.TOTAL_PACKAGES_FDEG == 0:
        G.FDEG_SORT_TIME = 0
        G.FDEG_PALLETS = 0
    if G.TOTAL_PACKAGES_FDE == 0:
        G.FDE_SORT_TIME = 0
        G.FDE_PALLETS = 0

    G.USPS_LINEHAUL_A_PACKAGES = sum_packages_by_type(linehaul_A_df, 'USPS')
    G.UPSN_LINEHAUL_A_PACKAGES = sum_packages_by_type(linehaul_A_df, 'UPSN')
    G.FDEG_LINEHAUL_A_PACKAGES = sum_packages_by_type(linehaul_A_df, 'FDEG')
    G.FDE_LINEHAUL_A_PACKAGES = sum_packages_by_type(linehaul_A_df, 'FDE')
    G.TLMD_LINEHAUL_A_PACKAGES = sum_packages_by_type(linehaul_A_df, 'TLMD')

    G.USPS_LINEHAUL_B_PACKAGES = sum_packages_by_type(linehaul_B_df, 'USPS')
    G.UPSN_LINEHAUL_B_PACKAGES = sum_packages_by_type(linehaul_B_df, 'UPSN')
    G.FDEG_LINEHAUL_B_PACKAGES = sum_packages_by_type(linehaul_B_df, 'FDEG')
    G.FDE_LINEHAUL_B_PACKAGES = sum_packages_by_type(linehaul_B_df, 'FDE')
    G.TLMD_LINEHAUL_B_PACKAGES = sum_packages_by_type(linehaul_B_df, 'TLMD')

    G.USPS_LINEHAUL_C_PACKAGES = sum_packages_by_type(linehaul_C_df, 'USPS')
    G.UPSN_LINEHAUL_C_PACKAGES = sum_packages_by_type(linehaul_C_df, 'UPSN')
    G.FDEG_LINEHAUL_C_PACKAGES = sum_packages_by_type(linehaul_C_df, 'FDEG')
    G.FDE_LINEHAUL_C_PACKAGES = sum_packages_by_type(linehaul_C_df, 'FDE')
    G.TLMD_LINEHAUL_C_PACKAGES = sum_packages_by_type(linehaul_C_df, 'TLMD')

    G.TLMD_LINEHAUL_TFC_PACKAGES = sum_packages_by_type(linehaul_TFC_df, 'TLMD')
    
    packages_flat = day_pallets['packages'].explode()
    packages_df = pd.DataFrame(packages_flat.tolist(), columns=['package_tracking_number', 'scac', 'Partition'])
    partition_counts_2 = packages_df.groupby('Partition')['package_tracking_number'].count()
    partition1_count = partition_counts_2.get('1', 0)
    partition2_count = partition_counts_2.get('2', 0)
    partition3AB_count = partition_counts_2.get('3AB', 0)
    partition3C_count = partition_counts_2.get('3C', 0)


    # Display the counts

    G.TLMD_PARTITION_1_PACKAGES = partition1_count
    G.TLMD_PARTITION_2_PACKAGES = partition2_count
    G.TLMD_PARTITION_3AB_PACKAGES = partition3AB_count
    G.TLMD_PARTITION_3C_PACKAGES = partition3C_count



    partition_1_packages = G.TLMD_PARTITION_1_PACKAGES
    partition_2_packages = G.TLMD_PARTITION_2_PACKAGES
    partition_3_packages =  G.TLMD_PARTITION_3AB_PACKAGES + G.TLMD_PARTITION_3C_PACKAGES
    partition_3AB_packages_actual = G.TLMD_PARTITION_3AB_PACKAGES




    if partition_1_packages + partition_2_packages + partition_3_packages != total_packages_TLMD:
        partition_3_packages = total_packages_TLMD - partition_1_packages - partition_2_packages

    if partition_3AB_packages_actual < 0:
        raise ValueError("Value for partition 3AB cannot be negative!")
    partition_3C_packages_actual =  G.TLMD_LINEHAUL_C_PACKAGES
    
    partition_1_pallets = math.ceil(partition_1_packages / G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
    partition_2_pallets = math.ceil(partition_2_packages / G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
    partition_3AB_pallets = math.ceil(partition_3AB_packages_actual / G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
    partition_3C_pallets = math.ceil(partition_3C_packages_actual / G.TLMD_PARTITION_PALLET_MAX_PACKAGES)
    #print(f'Partition 1 pallets: {partition_1_pallets}')
    #print(f'Partition 2 pallets: {partition_2_pallets}')
    #print(f'Partition 3AB pallets: {partition_3AB_pallets}')
    #print(f'Partition 3C pallets: {partition_3C_pallets}')


    G.TOTAL_PALLETS_TLMD = partition_1_pallets + partition_2_pallets + partition_3AB_pallets + partition_3C_pallets
    G.TLMD_PARTITION_1_PALLETS = partition_1_pallets
    G.TLMD_PARTITION_2_PALLETS = partition_2_pallets
    G.TLMD_PARTITION_3AB_PALLETS = partition_3AB_pallets
    G.TLMD_PARTITION_3C_PALLETS = partition_3C_pallets


    env = simpy.Environment()

    current_resources = {'tm_pit_unload': simpy.Resource(env, capacity=night_tm_pit_unload), 
                        'tm_pit_induct': simpy.PriorityResource(env, capacity=night_tm_pit_induct), 
                        'tm_pit_induct_stage': simpy.Resource(env, capacity=1),
                        'tm_nonpit_split': simpy.Resource(env, capacity=night_tm_nonpit_split), 
                        'tm_nonpit_NC': simpy.PriorityResource(env, capacity=night_tm_nonpit_NC), 
                        'tm_nonpit_buffer': simpy.PriorityResource(env, capacity=night_tm_nonpit_buffer),
                        'tm_TLMD_induct': simpy.PriorityResource(env, capacity=night_tm_TLMD_induct),
                        'tm_TLMD_picker': simpy.Resource(env, capacity=night_tm_TLMD_picker),
                        'tm_TLMD_sort': simpy.Resource(env, capacity=night_tm_TLMD_sort),
                        'tm_TLMD_stage': simpy.Resource(env, capacity=night_tm_TLMD_stage)}
    
    sortation_center = Sortation_Center_Original(env, 
                                        day_pallets, 
                                        USPS_Fluid_Status,
                                        UPSN_Fluid_Status,
                                        FDEG_Fluid_Status,
                                        FDE_Fluid_Status,
                                        current_resources,
                                        var_status,
                                        process_variance,
                                        TFC_status
                                        )

    # Start tracking metrics and schedule arrivals
    env.process(sortation_center.track_metrics())
    sortation_center.schedule_arrivals()
    env.process(linehaul_C_arrival(env,sortation_center))
    env.process(TFC_arrival(env,sortation_center))
    env.process(between_shift(env, sortation_center))
    env.process(break_1(env, sortation_center))
    env.process(break_2(env, sortation_center))
    env.process(break_3(env, sortation_center))
    env.process(break_4(env, sortation_center))
    #env.process(break_preshift(env, sortation_center))

    # for start, end in unavailable_periods:
    #     env.process(make_resources_unavailable(env, sortation_center, start, end))

    env.process(manage_resources(env, 
                                 sortation_center,
                                 current_resources,
                                 night_tm_pit_unload, 
                                 night_tm_pit_induct, 
                                 night_tm_nonpit_split, 
                                 night_tm_nonpit_NC, 
                                 night_tm_nonpit_buffer,
                                 night_tm_TLMD_induct,
                                 night_tm_TLMD_induct_stage,
                                 night_tm_TLMD_picker,
                                 night_tm_TLMD_sort,
                                 night_tm_TLMD_stage,
                                 day_tm_pit_unload,
                                 day_tm_pit_induct,
                                 day_tm_nonpit_split,
                                 day_tm_nonpit_NC,
                                 day_tm_nonpit_buffer,
                                 day_tm_TLMD_induct,
                                 day_tm_TLMD_induct_stage,
                                 day_tm_TLMD_picker,
                                 day_tm_TLMD_sort,
                                 day_tm_TLMD_stage))

   

    return env, sortation_center

#####################################################################################

def Simulation_Machine(predict,
                       night_total_tm,
                       day_total_tm,
                       night_tm_pit_unload, 
                        night_tm_pit_induct, 
                        night_tm_nonpit_split, 
                        night_tm_nonpit_NC, 
                        night_tm_nonpit_buffer,
                        night_tm_TLMD_induct,
                        night_tm_TLMD_induct_stage,
                        night_tm_TLMD_picker,
                        night_tm_TLMD_sort, 
                        night_tm_TLMD_stage,
                        day_tm_pit_unload,
                        day_tm_pit_induct,
                        day_tm_nonpit_split,
                        day_tm_nonpit_NC,
                        day_tm_nonpit_buffer,
                        day_tm_TLMD_induct,
                        day_tm_TLMD_induct_stage,
                        day_tm_TLMD_picker,
                        day_tm_TLMD_sort,
                        day_tm_TLMD_stage,
                        USPS_Fluid_Status,
                        UPSN_Fluid_Status,
                        FDEG_Fluid_Status,
                        FDE_Fluid_Status,
                        var_05,
                        var_10,
                        var_15,
                        var_20,
                        var_25,
                        var_30,
                        var_35,
                        var_40,
                        TFC_status
                        ):
    df_pallets, df_pallets_2, df_package_distribution, TFC_arrival_minutes, partition_1_count, partition_2_count, partition_3AB_count, partition_3C_count, partition_counts = sg_c.simulation_generator(predict, [0.40, 0.30, 0.30])
#df_pallets = pd.read_csv('8-15-24_packages.csv')
    while partition_1_count + partition_2_count + partition_3AB_count + partition_3C_count != sum(partition_counts.values()):
        df_pallets, df_pallets_2, df_package_distribution, TFC_arrival_minutes, partition_1_count, partition_2_count, partition_3AB_count, partition_3C_count, partition_counts = sg_c.simulation_generator(predict, [0.40, 0.30, 0.30])

    pallet_info = df_pallets.groupby('Pallet').agg(
        num_packages=('package_tracking_number', 'count'),
        earliest_arrival=('pkg_received_utc_ts', 'min'),
        packages=('package_tracking_number', lambda x: list(zip(x, df_pallets.loc[x.index, 'scac'], df_pallets.loc[x.index, 'Partition']))),
        linehaul=('Linehaul', 'first'),
    ).reset_index()
    pallet_info.to_csv('pallet_info.csv')

    pallet_info_2 = df_pallets_2.groupby('Pallet').agg(
        num_packages=('package_tracking_number', 'count'),
        earliest_arrival=('pkg_received_utc_ts', 'min'),
        packages=('package_tracking_number', lambda x: list(zip(x, df_pallets.loc[x.index, 'scac'], df_pallets.loc[x.index, 'Partition']))),
        linehaul=('Linehaul', 'first'),
    ).reset_index()
    pallet_info_2.to_csv('pallet_info_2.csv')

    G.TLMD_PARTITION_1_PACKAGES = partition_1_count
    G.TLMD_PARTITION_2_PACKAGES = partition_2_count
    G.TLMD_PARTITION_3AB_PACKAGES = partition_3AB_count
    G.TLMD_PARTITION_3C_PACKAGES = partition_3C_count

    if night_tm_pit_unload + night_tm_pit_induct + night_tm_nonpit_split + night_tm_nonpit_NC + night_tm_nonpit_buffer > night_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')   
    
    if night_tm_TLMD_induct + night_tm_TLMD_picker + night_tm_TLMD_sort + night_tm_TLMD_induct_stage  >  night_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')

    if night_tm_TLMD_stage > night_total_tm > night_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')
    
    
    if day_tm_pit_unload + day_tm_pit_induct + day_tm_nonpit_split + day_tm_nonpit_NC + day_tm_nonpit_buffer > day_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')   
    
    if day_tm_TLMD_induct + day_tm_TLMD_picker + day_tm_TLMD_sort + day_tm_TLMD_induct_stage  >  day_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')

    if day_tm_TLMD_stage > day_total_tm > day_total_tm:
        raise ValueError('Total number of TMs exceeds the limit')
    
    USPS_Fluid_Status = False
    UPSN_Fluid_Status = False
    FDEG_Fluid_Status = False
    FDE_Fluid_Status = False
    
 

    var_status = False
    TFC_status = True
    process_variance = 0
    # Setup inbound induct simulation
    env, sortation_center = setup_simulation(
                                             pallet_info, 
                                             night_tm_pit_unload, 
                                             night_tm_pit_induct, 
                                             night_tm_nonpit_split, 
                                             night_tm_nonpit_NC, 
                                             night_tm_nonpit_buffer,
                                             night_tm_TLMD_induct,
                                             night_tm_TLMD_induct_stage,
                                             night_tm_TLMD_picker,
                                             night_tm_TLMD_sort, 
                                             night_tm_TLMD_stage,
                                             day_tm_pit_unload,
                                             day_tm_pit_induct,
                                             day_tm_nonpit_split,
                                             day_tm_nonpit_NC,
                                             day_tm_nonpit_buffer,
                                             day_tm_TLMD_induct,
                                             day_tm_TLMD_induct_stage,
                                             day_tm_TLMD_picker,
                                             day_tm_TLMD_sort,
                                             day_tm_TLMD_stage,
                                             USPS_Fluid_Status,
                                             UPSN_Fluid_Status,
                                             FDEG_Fluid_Status,
                                             FDE_Fluid_Status,
                                             var_status,
                                             process_variance,
                                             TFC_status
                                             )   


    # Run inbound induct simulation
    #print("Begin Process")
    env.run(until=1410)
    final_packages = packages_to_dataframe(sortation_center.packages)
    #print("End Process")
    #print(len(G.TLMD_STAGED_PACKAGES))
    
    #plot_metrics(sortation_center.metrics)
    results = {
    # Total Packages
    "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
    "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
    "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
    # TLMD Partition Packages
    "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
    "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
    "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
    "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
    # Sorted Packages
    "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
    # Linehaul Totals
    "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
    "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
    "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
    # Linehaul by Carrier
    "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
    "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
    "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
    "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
    "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
    "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
    "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
    "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
    "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
    "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
    "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
    "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
    "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
    "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
    "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
    "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
    # Induction Times
    "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
    "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
    'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
    # Partition Sort Times
    "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
    "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
    "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
    "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
    # Sort Times by Carrier
    "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
    "USPS_SORT_TIME": G.USPS_SORT_TIME,
    "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
    "FDE_SORT_TIME": G.FDE_SORT_TIME,
    # Pallet Counts
    "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
    "UPSN_PALLETS": G.UPSN_PALLETS,
    "USPS_PALLETS": G.USPS_PALLETS,
    "FDEG_PALLETS": G.FDEG_PALLETS,
    "FDE_PALLETS": G.FDE_PALLETS,
    # Passed-Over Pallets
    "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
}

   
    G.TOTAL_PACKAGES = None  # Total packages to be processed
    G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
    G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
    G.TLMD_AB_INDUCT_TIME = None
    G.TLMD_C_INDUCT_TIME = None
    G.TLMD_STAGED_PACKAGES = None
    G.TLMD_PARTITION_1_PACKAGES = None
    G.TLMD_PARTITION_2_PACKAGES = None
    G.TLMD_PARTITION_3AB_PACKAGES = None
    G.TLMD_PARTITION_3C_PACKAGES = None
    G.TOTAL_PALLETS_TLMD = None
    G.TLMD_PARTITION_1_SORT_TIME = None
    G.TLMD_PARTITION_2_SORT_TIME = None 
    G.TLMD_PARTITION_3AB_SORT_TIME = None
    G.TLMD_PARTITION_3_SORT_TIME = None
    G.TLMD_SORTED_PACKAGES = None
    G.TLMD_PARTITION_1_CART_STAGE_TIME = None
    G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
    G.TLMD_PARTITION_3_CART_STAGE_TIME = None
    G.TLMD_OUTBOUND_PACKAGES = None
    G.I=1
    G.J=1
    G.K=1
    G.PASSED_OVER_PACKAGES = None

    G.TOTAL_LINEHAUL_A_PACKAGES = None
    G.TOTAL_LINEHAUL_B_PACKAGES = None
    G.TOTAL_LINEHAUL_C_PACKAGES = None

    G.USPS_LINEHAUL_A_PACKAGES = None
    G.USPS_LINEHAUL_B_PACKAGES = None
    G.USPS_LINEHAUL_C_PACKAGES = None

    G.UPSN_LINEHAUL_A_PACKAGES = None
    G.UPSN_LINEHAUL_B_PACKAGES = None
    G.UPSN_LINEHAUL_C_PACKAGES = None

    G.FDEG_LINEHAUL_A_PACKAGES = None
    G.FDEG_LINEHAUL_B_PACKAGES = None
    G.FDEG_LINEHAUL_C_PACKAGES = None

    G.FDE_LINEHAUL_A_PACKAGES = None
    G.FDE_LINEHAUL_B_PACKAGES = None
    G.FDE_LINEHAUL_C_PACKAGES = None

    G.TLMD_LINEHAUL_A_PACKAGES = None
    G.TLMD_LINEHAUL_B_PACKAGES = None
    G.TLMD_LINEHAUL_C_PACKAGES = None

    G.TLMD_LINEHAUL_TFC_PACKAGES=None

    G.LINEHAUL_C_TIME = None
    G.LINEHAUL_TFC_TIME = None

    G.TOTAL_PACKAGES_UPSN = None
    G.TOTAL_PACKAGES_USPS = None
    G.TOTAL_PACKAGES_FDEG = None
    G.TOTAL_PACKAGES_FDE = None
    G.UPSN_PALLETS = None
    G.USPS_PALLETS = None
    G.FDEG_PALLETS = None
    G.FDE_PALLETS = None
    G.UPSN_SORT_TIME = None
    G.USPS_SORT_TIME = None
    G.FDEG_SORT_TIME = None
    G.FDE_SORT_TIME = None

    G.TOTAL_CARTS_TLMD = None
    G.PASSED_OVER_PACKAGES = None

    del sortation_center    
    gc.collect()
    

    TFC_status = False
    process_variance = 0
    # Setup inbound induct simulation
    env, sortation_center = setup_simulation(
                                             pallet_info_2, 
                                             night_tm_pit_unload, 
                                             night_tm_pit_induct, 
                                             night_tm_nonpit_split, 
                                             night_tm_nonpit_NC, 
                                             night_tm_nonpit_buffer,
                                             night_tm_TLMD_induct,
                                             night_tm_TLMD_induct_stage,
                                             night_tm_TLMD_picker,
                                             night_tm_TLMD_sort, 
                                             night_tm_TLMD_stage,
                                             day_tm_pit_unload,
                                             day_tm_pit_induct,
                                             day_tm_nonpit_split,
                                             day_tm_nonpit_NC,
                                             day_tm_nonpit_buffer,
                                             day_tm_TLMD_induct,
                                             day_tm_TLMD_induct_stage,
                                             day_tm_TLMD_picker,
                                             day_tm_TLMD_sort,
                                             day_tm_TLMD_stage,
                                             USPS_Fluid_Status,
                                             UPSN_Fluid_Status,
                                             FDEG_Fluid_Status,
                                             FDE_Fluid_Status,
                                             var_status,
                                             process_variance,
                                             TFC_status
                                             )   


    # Run inbound induct simulation
    #print("Begin Process")
    env.run(until=1410)
    final_packages_2 = packages_to_dataframe(sortation_center.packages)
    #print("End Process")
    #print(len(G.TLMD_STAGED_PACKAGES))
    #plot_metrics(sortation_center.metrics)

    results_pallet = {
    # Total Packages
    "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
    "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
    "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
    # TLMD Partition Packages
    "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
    "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
    "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
    "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
    # Sorted Packages
    "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
    # Linehaul Totals
    "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
    "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
    "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
    # Linehaul by Carrier
    "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
    "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
    "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
    "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
    "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
    "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
    "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
    "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
    "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
    "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
    "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
    "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
    "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
    "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
    "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
    "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
    # Induction Times
    "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
    "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
    'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
    # Partition Sort Times
    "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
    "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
    "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
    "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
    # Sort Times by Carrier
    "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
    "USPS_SORT_TIME": G.USPS_SORT_TIME,
    "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
    "FDE_SORT_TIME": G.FDE_SORT_TIME,
    # Pallet Counts
    "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
    "UPSN_PALLETS": G.UPSN_PALLETS,
    "USPS_PALLETS": G.USPS_PALLETS,
    "FDEG_PALLETS": G.FDEG_PALLETS,
    "FDE_PALLETS": G.FDE_PALLETS,
    # Passed-Over Pallets
    "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
}

   
    G.TOTAL_PACKAGES = None  # Total packages to be processed
    G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
    G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
    G.TLMD_AB_INDUCT_TIME = None
    G.TLMD_C_INDUCT_TIME = None
    G.TLMD_STAGED_PACKAGES = None
    G.TLMD_PARTITION_1_PACKAGES = None
    G.TLMD_PARTITION_2_PACKAGES = None
    G.TLMD_PARTITION_3AB_PACKAGES = None
    G.TLMD_PARTITION_3C_PACKAGES = None
    G.TOTAL_PALLETS_TLMD = None
    G.TLMD_PARTITION_1_SORT_TIME = None
    G.TLMD_PARTITION_2_SORT_TIME = None 
    G.TLMD_PARTITION_3AB_SORT_TIME = None
    G.TLMD_PARTITION_3_SORT_TIME = None
    G.TLMD_SORTED_PACKAGES = None
    G.TLMD_PARTITION_1_CART_STAGE_TIME = None
    G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
    G.TLMD_PARTITION_3_CART_STAGE_TIME = None
    G.TLMD_OUTBOUND_PACKAGES = None
    G.I=1
    G.J=1
    G.K=1
    G.PASSED_OVER_PACKAGES = None

    G.TOTAL_LINEHAUL_A_PACKAGES = None
    G.TOTAL_LINEHAUL_B_PACKAGES = None
    G.TOTAL_LINEHAUL_C_PACKAGES = None

    G.USPS_LINEHAUL_A_PACKAGES = None
    G.USPS_LINEHAUL_B_PACKAGES = None
    G.USPS_LINEHAUL_C_PACKAGES = None

    G.UPSN_LINEHAUL_A_PACKAGES = None
    G.UPSN_LINEHAUL_B_PACKAGES = None
    G.UPSN_LINEHAUL_C_PACKAGES = None

    G.FDEG_LINEHAUL_A_PACKAGES = None
    G.FDEG_LINEHAUL_B_PACKAGES = None
    G.FDEG_LINEHAUL_C_PACKAGES = None

    G.FDE_LINEHAUL_A_PACKAGES = None
    G.FDE_LINEHAUL_B_PACKAGES = None
    G.FDE_LINEHAUL_C_PACKAGES = None

    G.TLMD_LINEHAUL_A_PACKAGES = None
    G.TLMD_LINEHAUL_B_PACKAGES = None
    G.TLMD_LINEHAUL_C_PACKAGES = None

    G.TLMD_LINEHAUL_TFC_PACKAGES=None

    G.LINEHAUL_C_TIME = None
    G.LINEHAUL_TFC_TIME = None

    G.TOTAL_PACKAGES_UPSN = None
    G.TOTAL_PACKAGES_USPS = None
    G.TOTAL_PACKAGES_FDEG = None
    G.TOTAL_PACKAGES_FDE = None
    G.UPSN_PALLETS = None
    G.USPS_PALLETS = None
    G.FDEG_PALLETS = None
    G.FDE_PALLETS = None
    G.UPSN_SORT_TIME = None
    G.USPS_SORT_TIME = None
    G.FDEG_SORT_TIME = None
    G.FDE_SORT_TIME = None

    G.TOTAL_CARTS_TLMD = None
    G.PASSED_OVER_PACKAGES = None

    del sortation_center    
    gc.collect()


    ##########################################
    var_status = True
    TFC_status = True

    if var_05:
        process_variance = 0.5
        # Setup inbound induct simulation
        env, sortation_center = setup_simulation(
                                                pallet_info, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   


        # Run inbound induct simulation
        #print("Begin Process")
        env.run(until=1410)
        #print("End Process")
        #print(len(G.TLMD_STAGED_PACKAGES))

        #print("Variability")
        #plot_metrics(sortation_center.metrics)

        results_var_05 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }

    
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()
    
        TFC_status = False
        process_variance = 0.5
        # Setup inbound induct simulation
        env, sortation_center = setup_simulation(
                                                pallet_info_2, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   


        # Run inbound induct simulation
        #print("Begin Process")
        env.run(until=1410)
        #print("End Process")
        #print(len(G.TLMD_STAGED_PACKAGES))

        #print("Variability")
        #plot_metrics(sortation_center.metrics)

        results_pallet_var_05 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }

    
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()
    
    else:
        results_var_05 = None
        results_pallet_var_05 = None

    if var_10:
        process_variance = .1
        TFC_status = True
        env, sortation_center = setup_simulation(
                                                 pallet_info, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   
        
        env.run(until=1410)
        #print("No Variability")
        #plot_metrics(sortation_center.metrics)

        results_var_10 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }
        
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()

        TFC_status = False
        process_variance = .1
        env, sortation_center = setup_simulation(
                                                 pallet_info_2, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   
        
        env.run(until=1410)
        #print("No Variability")
        #plot_metrics(sortation_center.metrics)

        results_pallet_var_10 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }
        
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()


    else:
        results_var_10 = None
        results_pallet_var_10 = None

    if var_15:
        TFC_status = True
        process_variance = 0.15
        # Setup inbound induct simulation
        env, sortation_center = setup_simulation( pallet_info, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   


        # Run inbound induct simulation
        #print("Begin Process")
        env.run(until=1410)
        #print("End Process")
        #print(len(G.TLMD_STAGED_PACKAGES))

        #print("Variability")
        #plot_metrics(sortation_center.metrics)

        results_var_15 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }

    
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()

    else:
        results_var_15 = None


    if var_20:

        process_variance = 0.2
        TFC_status = True
        env, sortation_center = setup_simulation(
                                                pallet_info, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   
        
        env.run(until=1410)
        #print("No Variability")
        #plot_metrics(sortation_center.metrics)

        results_var_20 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }
        
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()

        process_variance = 0.2
        TFC_status = False
        env, sortation_center = setup_simulation(
                                                pallet_info_2, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   
        
        env.run(until=1410)

        results_pallet_var_20 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }
        
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()

    else:
        results_var_20 = None
        results_pallet_var_20 = None

    if var_25:
        TFC_status = True
        process_variance = 0.25
        env, sortation_center = setup_simulation(
                                                pallet_info, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   
        
        env.run(until=1410)
        #print("No Variability")
        #plot_metrics(sortation_center.metrics)

        results_var_25 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets

        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }
        
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()

    
    else:
        results_var_25 = None

    if var_30:
        TFC_status = True
        process_variance = 0.3
        env, sortation_center = setup_simulation(
                                                pallet_info, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   
        
        env.run(until=1410)
        #print("No Variability")
        #plot_metrics(sortation_center.metrics)

        results_var_30 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }
        
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()

        TFC_status = False
        process_variance = 0.3
        env, sortation_center = setup_simulation(
                                                pallet_info_2, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   
        
        env.run(until=1410)
        #print("No Variability")
        #plot_metrics(sortation_center.metrics)

        results_pallet_var_30 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }
        
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()
    
    else:
        results_var_30 = None
        results_pallet_var_30 = None

    if var_35 == True:
        TFC_status = True
        process_variance = 0.35
        env, sortation_center = setup_simulation(
                                                pallet_info, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   
        
        env.run(until=1410)
        #print("No Variability")
        #plot_metrics(sortation_center.metrics)

        results_var_35 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }
        
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()
        
    else:
        results_var_35 = None

    if var_40:
        TFC_status = True
        process_variance = 0.4
        env, sortation_center = setup_simulation(
                                                pallet_info, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   
        
        env.run(until=1410)
        #print("No Variability")
        #plot_metrics(sortation_center.metrics)

        results_var_40 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }
        
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()
        TFC_status = False
        process_variance = 0.4
        env, sortation_center = setup_simulation(
                                                pallet_info_2, 
                                                night_tm_pit_unload, 
                                                night_tm_pit_induct, 
                                                night_tm_nonpit_split, 
                                                night_tm_nonpit_NC, 
                                                night_tm_nonpit_buffer,
                                                night_tm_TLMD_induct,
                                                night_tm_TLMD_induct_stage,
                                                night_tm_TLMD_picker,
                                                night_tm_TLMD_sort, 
                                                night_tm_TLMD_stage,
                                                day_tm_pit_unload,
                                                day_tm_pit_induct,
                                                day_tm_nonpit_split,
                                                day_tm_nonpit_NC,
                                                day_tm_nonpit_buffer,
                                                day_tm_TLMD_induct,
                                                day_tm_TLMD_induct_stage,
                                                day_tm_TLMD_picker,
                                                day_tm_TLMD_sort,
                                                day_tm_TLMD_stage,
                                                USPS_Fluid_Status,
                                                UPSN_Fluid_Status,
                                                FDEG_Fluid_Status,
                                                FDE_Fluid_Status,
                                                var_status,
                                                process_variance,
                                                TFC_status
                                                )   
        
        env.run(until=1410)
        #print("No Variability")
        #plot_metrics(sortation_center.metrics)

        results_pallet_var_40 = {
        # Total Packages
        "TOTAL_PACKAGES": G.TOTAL_PACKAGES,
        "TOTAL_PACKAGES_TLMD": G.TOTAL_PACKAGES_TLMD,
        "TOTAL_PACKAGES_NC": G.TOTAL_PACKAGES_NC,
        # TLMD Partition Packages
        "TLMD_PARTITION_1_PACKAGES": G.TLMD_PARTITION_1_PACKAGES,
        "TLMD_PARTITION_2_PACKAGES": G.TLMD_PARTITION_2_PACKAGES,
        "TLMD_PARTITION_3AB_PACKAGES": G.TLMD_PARTITION_3AB_PACKAGES,
        "TLMD_PARTITION_3C_PACKAGES": G.TLMD_PARTITION_3C_PACKAGES,
        # Sorted Packages
        "TLMD_SORTED_PACKAGES": G.TLMD_SORTED_PACKAGES,
        # Linehaul Totals
        "TOTAL_LINEHAUL_A_PACKAGES": G.TOTAL_LINEHAUL_A_PACKAGES,
        "TOTAL_LINEHAUL_B_PACKAGES": G.TOTAL_LINEHAUL_B_PACKAGES,
        "TOTAL_LINEHAUL_C_PACKAGES": G.TOTAL_LINEHAUL_C_PACKAGES,
        # Linehaul by Carrier
        "USPS_LINEHAUL_A_PACKAGES": G.USPS_LINEHAUL_A_PACKAGES,
        "USPS_LINEHAUL_B_PACKAGES": G.USPS_LINEHAUL_B_PACKAGES,
        "USPS_LINEHAUL_C_PACKAGES": G.USPS_LINEHAUL_C_PACKAGES,
        "UPSN_LINEHAUL_A_PACKAGES": G.UPSN_LINEHAUL_A_PACKAGES,
        "UPSN_LINEHAUL_B_PACKAGES": G.UPSN_LINEHAUL_B_PACKAGES,
        "UPSN_LINEHAUL_C_PACKAGES": G.UPSN_LINEHAUL_C_PACKAGES,
        "FDEG_LINEHAUL_A_PACKAGES": G.FDEG_LINEHAUL_A_PACKAGES,
        "FDEG_LINEHAUL_B_PACKAGES": G.FDEG_LINEHAUL_B_PACKAGES,
        "FDEG_LINEHAUL_C_PACKAGES": G.FDEG_LINEHAUL_C_PACKAGES,
        "FDE_LINEHAUL_A_PACKAGES": G.FDE_LINEHAUL_A_PACKAGES,
        "FDE_LINEHAUL_B_PACKAGES": G.FDE_LINEHAUL_B_PACKAGES,
        "FDE_LINEHAUL_C_PACKAGES": G.FDE_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_A_PACKAGES": G.TLMD_LINEHAUL_A_PACKAGES,
        "TLMD_LINEHAUL_B_PACKAGES": G.TLMD_LINEHAUL_B_PACKAGES,
        "TLMD_LINEHAUL_C_PACKAGES": G.TLMD_LINEHAUL_C_PACKAGES,
        "TLMD_LINEHAUL_TFC_PACKAGES": G.TLMD_LINEHAUL_TFC_PACKAGES,
        # Induction Times
        "TLMD_AB_INDUCT_TIME": G.TLMD_AB_INDUCT_TIME,
        "TLMD_C_INDUCT_TIME": G.TLMD_C_INDUCT_TIME,
        'LINEHAUL_TFC_TIME': G.LINEHAUL_TFC_TIME,
        # Partition Sort Times
        "TLMD_PARTITION_1_SORT_TIME": G.TLMD_PARTITION_1_SORT_TIME,
        "TLMD_PARTITION_2_SORT_TIME": G.TLMD_PARTITION_2_SORT_TIME,
        "TLMD_PARTITION_3AB_SORT_TIME": G.TLMD_PARTITION_3AB_SORT_TIME,
        "TLMD_PARTITION_3_SORT_TIME": G.TLMD_PARTITION_3_SORT_TIME,
        # Sort Times by Carrier
        "UPSN_SORT_TIME": G.UPSN_SORT_TIME,
        "USPS_SORT_TIME": G.USPS_SORT_TIME,
        "FDEG_SORT_TIME": G.FDEG_SORT_TIME,
        "FDE_SORT_TIME": G.FDE_SORT_TIME,
        # Pallet Counts
        "TOTAL_PALLETS_TLMD": G.TOTAL_PALLETS_TLMD,
        "UPSN_PALLETS": G.UPSN_PALLETS,
        "USPS_PALLETS": G.USPS_PALLETS,
        "FDEG_PALLETS": G.FDEG_PALLETS,
        "FDE_PALLETS": G.FDE_PALLETS,
        # Passed-Over Pallets
        "PASSED_OVER_PACKAGES_TLMD": G.PASSED_OVER_PACKAGES,
    }
        
        G.TOTAL_PACKAGES = None  # Total packages to be processed
        G.TOTAL_PACKAGES_TLMD = None  # Total TLMD packages to be processed
        G.TOTAL_PACKAGES_NC = None  # Total National Carrier packages to be processed
        G.TLMD_AB_INDUCT_TIME = None
        G.TLMD_C_INDUCT_TIME = None
        G.TLMD_STAGED_PACKAGES = None
        G.TLMD_PARTITION_1_PACKAGES = None
        G.TLMD_PARTITION_2_PACKAGES = None
        G.TLMD_PARTITION_3AB_PACKAGES = None
        G.TLMD_PARTITION_3C_PACKAGES = None
        G.TOTAL_PALLETS_TLMD = None
        G.TLMD_PARTITION_1_SORT_TIME = None
        G.TLMD_PARTITION_2_SORT_TIME = None 
        G.TLMD_PARTITION_3AB_SORT_TIME = None
        G.TLMD_PARTITION_3_SORT_TIME = None
        G.TLMD_SORTED_PACKAGES = None
        G.TLMD_PARTITION_1_CART_STAGE_TIME = None
        G.TLMD_PARTITION_2_CART_STAGE_TIME = None 
        G.TLMD_PARTITION_3_CART_STAGE_TIME = None
        G.TLMD_OUTBOUND_PACKAGES = None
        G.I=1
        G.J=1
        G.K=1
        G.PASSED_OVER_PACKAGES = None

        G.TOTAL_LINEHAUL_A_PACKAGES = None
        G.TOTAL_LINEHAUL_B_PACKAGES = None
        G.TOTAL_LINEHAUL_C_PACKAGES = None

        G.USPS_LINEHAUL_A_PACKAGES = None
        G.USPS_LINEHAUL_B_PACKAGES = None
        G.USPS_LINEHAUL_C_PACKAGES = None

        G.UPSN_LINEHAUL_A_PACKAGES = None
        G.UPSN_LINEHAUL_B_PACKAGES = None
        G.UPSN_LINEHAUL_C_PACKAGES = None

        G.FDEG_LINEHAUL_A_PACKAGES = None
        G.FDEG_LINEHAUL_B_PACKAGES = None
        G.FDEG_LINEHAUL_C_PACKAGES = None

        G.FDE_LINEHAUL_A_PACKAGES = None
        G.FDE_LINEHAUL_B_PACKAGES = None
        G.FDE_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_A_PACKAGES = None
        G.TLMD_LINEHAUL_B_PACKAGES = None
        G.TLMD_LINEHAUL_C_PACKAGES = None

        G.TLMD_LINEHAUL_TFC_PACKAGES=None

        G.LINEHAUL_C_TIME = None
        G.LINEHAUL_TFC_TIME = None

        G.TOTAL_PACKAGES_UPSN = None
        G.TOTAL_PACKAGES_USPS = None
        G.TOTAL_PACKAGES_FDEG = None
        G.TOTAL_PACKAGES_FDE = None
        G.UPSN_PALLETS = None
        G.USPS_PALLETS = None
        G.FDEG_PALLETS = None
        G.FDE_PALLETS = None
        G.UPSN_SORT_TIME = None
        G.USPS_SORT_TIME = None
        G.FDEG_SORT_TIME = None
        G.FDE_SORT_TIME = None

        G.TOTAL_CARTS_TLMD = None
        G.PASSED_OVER_PACKAGES = None

        del sortation_center    
        gc.collect()

    else:
        results_var_40 = None
        results_pallet_var_40 = None


    return results, results_pallet, final_packages, final_packages_2, results_var_05, results_pallet_var_05, results_var_10, results_pallet_var_10, results_var_15, results_var_20, results_pallet_var_20, results_var_25, results_var_30, results_pallet_var_30, results_var_35, results_var_40, results_pallet_var_40,  df_package_distribution, TFC_arrival_minutes





