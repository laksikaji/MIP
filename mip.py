import pandas as pd
import pulp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== ฟังก์ชันคำนวณ FTHR ====================
def calculate_fthr(distance):
    """คำนวณ FTHR ตามตาราง 3.1 ในเอกสาร"""
    if distance <= 5:
        return 0.95
    elif distance <= 10:
        return 0.85
    elif distance <= 15:
        return 0.70
    elif distance <= 20:
        return 0.55
    else:
        return 0.40

# ==================== โหลดข้อมูล ====================
print("="*70)
print("Loading data files...")
print("="*70)

# อ่านไฟล์ CSV
matrix_df = pd.read_csv('matrix.csv')
dcs_df = pd.read_csv('dcs.csv')
demand_df = pd.read_csv('demand.csv')

# พารามิเตอร์
TRANSPORT_COST_PER_KM = 15  # บาท/กิโลเมตร (จากเอกสาร)

# คำนวณต้นทุนขนส่งและ FTHR
matrix_df['transport_cost'] = matrix_df['distance_km'] * TRANSPORT_COST_PER_KM
matrix_df['fthr'] = matrix_df['distance_km'].apply(calculate_fthr)

# สร้าง dictionaries สำหรับใช้ในโมเดล
customers = demand_df['customer_id'].unique()
dcs = dcs_df['dc_id'].unique()

demand = dict(zip(demand_df['customer_id'], demand_df['demand']))
capacity = dict(zip(dcs_df['dc_id'], dcs_df['capacity']))
fixed_cost = dict(zip(dcs_df['dc_id'], dcs_df['fixed_cost']))

# สร้าง nested dict สำหรับ distance, cost, fthr
distance = {}
transport_cost = {}
fthr = {}

for _, row in matrix_df.iterrows():
    i, j = row['customer_id'], row['dc_id']
    distance[(i, j)] = row['distance_km']
    transport_cost[(i, j)] = row['transport_cost']
    fthr[(i, j)] = row['fthr']

print(f"[OK] Loaded {len(customers)} customers")
print(f"[OK] Loaded {len(dcs)} distribution centers")
print(f"[OK] Transport cost rate: {TRANSPORT_COST_PER_KM} THB/km")
print()

# ==================== แบบจำลอง 1: Basic MIP ====================
print("="*70)
print("MODEL 1: BASIC MIP (without FTHR)")
print("="*70)

start_time = datetime.now()

# สร้างโมเดล
model_basic = pulp.LpProblem("Basic_MIP_DC_Selection", pulp.LpMinimize)

# ตัวแปรการตัดสินใจ
x_basic = pulp.LpVariable.dicts("assign", 
                                 [(i, j) for i in customers for j in dcs],
                                 cat='Binary')
y_basic = pulp.LpVariable.dicts("open", dcs, cat='Binary')

# ฟังก์ชันวัตถุประสงค์: Minimize Total Cost
model_basic += (
    pulp.lpSum([fixed_cost[j] * y_basic[j] for j in dcs]) +  # Fixed cost
    pulp.lpSum([transport_cost[(i, j)] * x_basic[(i, j)] 
                for i in customers for j in dcs])  # Transport cost
), "Total_Cost"

# ข้อจำกัด 1: แต่ละลูกค้าต้องถูกจัดสรรให้ศูนย์เดียว
for i in customers:
    model_basic += (
        pulp.lpSum([x_basic[(i, j)] for j in dcs]) == 1,
        f"Customer_{i}_Assignment"
    )

# ข้อจำกัด 2: ความจุของศูนย์
for j in dcs:
    model_basic += (
        pulp.lpSum([demand[i] * x_basic[(i, j)] for i in customers]) 
        <= capacity[j] * y_basic[j],
        f"DC_{j}_Capacity"
    )

# ข้อจำกัด 3: ลูกค้าต้องถูกจัดสรรให้ศูนย์ที่เปิดเท่านั้น
for i in customers:
    for j in dcs:
        model_basic += (
            x_basic[(i, j)] <= y_basic[j],
            f"Linking_{i}_{j}"
        )

# แก้ปัญหา
model_basic.solve(pulp.PULP_CBC_CMD(msg=0))

basic_time = (datetime.now() - start_time).total_seconds()

# ผลลัพธ์
basic_status = pulp.LpStatus[model_basic.status]
basic_total_cost = pulp.value(model_basic.objective)
basic_opened_dcs = [j for j in dcs if pulp.value(y_basic[j]) == 1]
basic_assignments = {i: j for i in customers for j in dcs 
                     if pulp.value(x_basic[(i, j)]) == 1}

# คำนวณ metrics
basic_total_distance = sum(distance[(i, basic_assignments[i])] for i in customers)
basic_avg_distance = basic_total_distance / len(customers)
basic_avg_fthr = sum(fthr[(i, basic_assignments[i])] for i in customers) / len(customers)

print(f"Status: {basic_status}")
print(f"Total Cost: {basic_total_cost:,.2f} THB")
print(f"Opened DCs: {sorted(basic_opened_dcs)}")
print(f"Number of DCs: {len(basic_opened_dcs)}")
print(f"Total Distance: {basic_total_distance:.2f} km")
print(f"Average Distance: {basic_avg_distance:.2f} km")
print(f"Average FTHR: {basic_avg_fthr:.4f} ({basic_avg_fthr*100:.2f}%)")
print(f"Computation Time: {basic_time:.2f} seconds")
print()

# ==================== แบบจำลอง 2: FTHR-MIP ====================
print("="*70)
print("MODEL 2: FTHR-MIP (with First-Time Hit Rate)")
print("="*70)

start_time = datetime.now()

# สร้างโมเดล
model_fthr = pulp.LpProblem("FTHR_MIP_DC_Selection", pulp.LpMinimize)

# ตัวแปรการตัดสินใจ
x_fthr = pulp.LpVariable.dicts("assign", 
                                [(i, j) for i in customers for j in dcs],
                                cat='Binary')
y_fthr = pulp.LpVariable.dicts("open", dcs, cat='Binary')

# ฟังก์ชันวัตถุประสงค์: Minimize Total Cost + Expected Re-delivery Cost
model_fthr += (
    pulp.lpSum([fixed_cost[j] * y_fthr[j] for j in dcs]) +  # Fixed cost
    pulp.lpSum([transport_cost[(i, j)] * x_fthr[(i, j)] 
                for i in customers for j in dcs]) +  # Transport cost
    pulp.lpSum([transport_cost[(i, j)] * (1 - fthr[(i, j)]) * x_fthr[(i, j)]
                for i in customers for j in dcs])  # Expected re-delivery cost
), "Total_Cost_with_FTHR"

# ข้อจำกัด 1: แต่ละลูกค้าต้องถูกจัดสรรให้ศูนย์เดียว
for i in customers:
    model_fthr += (
        pulp.lpSum([x_fthr[(i, j)] for j in dcs]) == 1,
        f"Customer_{i}_Assignment"
    )

# ข้อจำกัด 2: ความจุของศูนย์
for j in dcs:
    model_fthr += (
        pulp.lpSum([demand[i] * x_fthr[(i, j)] for i in customers]) 
        <= capacity[j] * y_fthr[j],
        f"DC_{j}_Capacity"
    )

# ข้อจำกัด 3: ลูกค้าต้องถูกจัดสรรให้ศูนย์ที่เปิดเท่านั้น
for i in customers:
    for j in dcs:
        model_fthr += (
            x_fthr[(i, j)] <= y_fthr[j],
            f"Linking_{i}_{j}"
        )

# แก้ปัญหา
model_fthr.solve(pulp.PULP_CBC_CMD(msg=0))

fthr_time = (datetime.now() - start_time).total_seconds()

# ผลลัพธ์
fthr_status = pulp.LpStatus[model_fthr.status]
fthr_total_cost = pulp.value(model_fthr.objective)
fthr_opened_dcs = [j for j in dcs if pulp.value(y_fthr[j]) == 1]
fthr_assignments = {i: j for i in customers for j in dcs 
                    if pulp.value(x_fthr[(i, j)]) == 1}

# คำนวณ metrics
fthr_total_distance = sum(distance[(i, fthr_assignments[i])] for i in customers)
fthr_avg_distance = fthr_total_distance / len(customers)
fthr_avg_fthr = sum(fthr[(i, fthr_assignments[i])] for i in customers) / len(customers)

# คำนวณต้นทุนจัดส่งซ้ำที่คาดหมาย
fthr_expected_redelivery = sum(
    transport_cost[(i, fthr_assignments[i])] * (1 - fthr[(i, fthr_assignments[i])])
    for i in customers
)

print(f"Status: {fthr_status}")
print(f"Total Cost: {fthr_total_cost:,.2f} THB")
print(f"Opened DCs: {sorted(fthr_opened_dcs)}")
print(f"Number of DCs: {len(fthr_opened_dcs)}")
print(f"Total Distance: {fthr_total_distance:.2f} km")
print(f"Average Distance: {fthr_avg_distance:.2f} km")
print(f"Average FTHR: {fthr_avg_fthr:.4f} ({fthr_avg_fthr*100:.2f}%)")
print(f"Expected Re-delivery Cost: {fthr_expected_redelivery:,.2f} THB")
print(f"Computation Time: {fthr_time:.2f} seconds")
print()

# ==================== สรุปเปรียบเทียบ ====================
print("="*70)
print("COMPARISON SUMMARY")
print("="*70)

comparison_data = {
    'Metric': [
        'Total Cost (THB)',
        'Number of Opened DCs',
        'Total Distance (km)',
        'Average Distance (km)',
        'Average FTHR (%)',
        'Computation Time (sec)'
    ],
    'Basic MIP': [
        f"{basic_total_cost:,.2f}",
        len(basic_opened_dcs),
        f"{basic_total_distance:.2f}",
        f"{basic_avg_distance:.2f}",
        f"{basic_avg_fthr*100:.2f}",
        f"{basic_time:.2f}"
    ],
    'FTHR-MIP': [
        f"{fthr_total_cost:,.2f}",
        len(fthr_opened_dcs),
        f"{fthr_total_distance:.2f}",
        f"{fthr_avg_distance:.2f}",
        f"{fthr_avg_fthr*100:.2f}",
        f"{fthr_time:.2f}"
    ],
    'Difference': [
        f"{fthr_total_cost - basic_total_cost:+,.2f} ({((fthr_total_cost/basic_total_cost-1)*100):+.2f}%)",
        f"{len(fthr_opened_dcs) - len(basic_opened_dcs):+d}",
        f"{fthr_total_distance - basic_total_distance:+.2f}",
        f"{fthr_avg_distance - basic_avg_distance:+.2f}",
        f"{(fthr_avg_fthr - basic_avg_fthr)*100:+.2f}",
        f"{fthr_time - basic_time:+.2f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))
print()

# ==================== Export ผลลัพธ์ ====================
print("="*70)
print("EXPORTING RESULTS")
print("="*70)

# 1. สรุปผลการเปรียบเทียบ
comparison_df.to_csv('comparison_summary.csv', index=False)
print("[OK] Saved: comparison_summary.csv")

# 2. รายละเอียดการจับคู่ลูกค้า-ศูนย์ (Basic MIP)
basic_detail = pd.DataFrame([
    {
        'customer_id': i,
        'assigned_dc': basic_assignments[i],
        'demand': demand[i],
        'distance_km': distance[(i, basic_assignments[i])],
        'fthr': fthr[(i, basic_assignments[i])],
        'transport_cost': transport_cost[(i, basic_assignments[i])]
    }
    for i in customers
])
basic_detail.to_csv('basic_mip_assignments.csv', index=False)
print("[OK] Saved: basic_mip_assignments.csv")

# 3. รายละเอียดการจับคู่ลูกค้า-ศูนย์ (FTHR-MIP)
fthr_detail = pd.DataFrame([
    {
        'customer_id': i,
        'assigned_dc': fthr_assignments[i],
        'demand': demand[i],
        'distance_km': distance[(i, fthr_assignments[i])],
        'fthr': fthr[(i, fthr_assignments[i])],
        'transport_cost': transport_cost[(i, fthr_assignments[i])],
        'expected_redelivery_cost': transport_cost[(i, fthr_assignments[i])] * (1 - fthr[(i, fthr_assignments[i])])
    }
    for i in customers
])
fthr_detail.to_csv('fthr_mip_assignments.csv', index=False)
print("[OK] Saved: fthr_mip_assignments.csv")

# 4. สรุปศูนย์ที่เปิดใช้งาน
dc_summary = pd.DataFrame([
    {
        'dc_id': j,
        'opened_basic': 1 if j in basic_opened_dcs else 0,
        'opened_fthr': 1 if j in fthr_opened_dcs else 0,
        'capacity': capacity[j],
        'fixed_cost': fixed_cost[j],
        'customers_basic': sum(1 for i in customers if basic_assignments[i] == j),
        'customers_fthr': sum(1 for i in customers if fthr_assignments[i] == j),
        'load_basic': sum(demand[i] for i in customers if basic_assignments[i] == j),
        'load_fthr': sum(demand[i] for i in customers if fthr_assignments[i] == j)
    }
    for j in dcs
])
dc_summary.to_csv('dc_summary.csv', index=False)
print("[OK] Saved: dc_summary.csv")

# 5. สรุปรายงานแบบเต็ม
with open('full_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("LAST-MILE DELIVERY OPTIMIZATION: BASIC MIP vs FTHR-MIP\n")
    f.write("="*70 + "\n\n")
    
    f.write("=== MODEL 1: BASIC MIP ===\n")
    f.write(f"Status: {basic_status}\n")
    f.write(f"Total Cost: {basic_total_cost:,.2f} THB\n")
    f.write(f"Opened DCs: {sorted(basic_opened_dcs)}\n")
    f.write(f"Average FTHR: {basic_avg_fthr*100:.2f}%\n\n")
    
    f.write("=== MODEL 2: FTHR-MIP ===\n")
    f.write(f"Status: {fthr_status}\n")
    f.write(f"Total Cost: {fthr_total_cost:,.2f} THB\n")
    f.write(f"Opened DCs: {sorted(fthr_opened_dcs)}\n")
    f.write(f"Average FTHR: {fthr_avg_fthr*100:.2f}%\n")
    f.write(f"Expected Re-delivery Cost: {fthr_expected_redelivery:,.2f} THB\n\n")
    
    f.write("=== COMPARISON ===\n")
    f.write(comparison_df.to_string(index=False))

print("[OK] Saved: full_report.txt")
print()

print("="*70)
print("[OK] ALL TASKS COMPLETED SUCCESSFULLY")
print("="*70)