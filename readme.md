# README

## Conference Information
- Conference: **ICDCIT**
- Year: **2026**
- Submission ID: **ICDCIT26 Submission 139**
- Title: **Fault-Tolerant Decentralized Distributed Asynchronous Federated Learning with Adaptive Termination Detection**
- Track: **Distributed Computing**

## Disclaimer
The code is provided for the submission process to enhance transparency.  
However, this release of the code is preliminary and the ability to compile it may be sensitive to specific environment settings.  
We plan to upload a stable and reproducible version if the paper is accepted, as well as the datasets produced in the process.

## Input Configuration

The program expects an input file named **`inputf.txt`** in the same directory.  
Its format is:

<no_of_clients> <no_of_machines><br>
<current_machine_ip><br>
<all_machines_ip_address_comma_separated><br>
<no_of_faults><br>
<client_id>,<round_number>,<no_of_msgs_before_crashing><br>
<client_id>,<round_number>,<no_of_msgs_before_crashing><br>
...(continued for No. of faults)

## Example
6 3<br>
10.0.0.3<br>
10.0.0.1,10.0.0.1,10.0.0.2,10.0.0.2,10.0.0.3,10.0.0.3<br>
5<br>
0,45,3<br>
1,47,4<br>
2,44,5<br>
3,39,2<br>
4,49,1<br>



### Explanation

1. `6 3` → 6 clients across 3 machines.  

2. `10.0.0.3` → Current machine’s IP.  

3. Client-to-IP mapping (from the 3rd line):  
   - Client 0 → `10.0.0.1`  
   - Client 1 → `10.0.0.1`  
   - Client 2 → `10.0.0.2`  
   - Client 3 → `10.0.0.2`  
   - Client 4 → `10.0.0.3`  
   - Client 5 → `10.0.0.3`  

4. `5` → Number of crash faults.  

5. Crashes defined:  
   - Client 0 crashes at round 45 after 3 broadcasts.  
   - Client 1 crashes at round 47 after 4 broadcasts.  
   - Client 2 crashes at round 44 after 5 broadcasts.  
   - Client 3 crashes at round 39 after 2 broadcasts.  
   - Client 4 crashes at round 49 after 1 broadcast.  

---

In short:  
- First 3 lines: federation setup (clients, machines, IPs).  
- Fourth line: number of crashes.  
- Next lines: crash details (which client, when, how it behaves before stopping).  
