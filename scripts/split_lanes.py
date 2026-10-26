import os
import sys
import time
import pysam

in_bam_path = "results/alignment/Control_rep1.sorted.bam"
out6_path = "results/alignment/ERR127306.sorted.bam"
out7_path = "results/alignment/ERR127307.sorted.bam"

print(f"Opening {in_bam_path}...")
in_bam = pysam.AlignmentFile(in_bam_path, "rb", threads=2)
out6 = pysam.AlignmentFile(out6_path, "wb", template=in_bam, threads=2)
out7 = pysam.AlignmentFile(out7_path, "wb", template=in_bam, threads=2)

count6 = 0
count7 = 0
count_other = 0

start_time = time.time()
last_report = start_time

for i, read in enumerate(in_bam):
    qname = read.query_name
    if qname.startswith("ERR127306"):
        out6.write(read)
        count6 += 1
    elif qname.startswith("ERR127307"):
        out7.write(read)
        count7 += 1
    else:
        count_other += 1

    if (i + 1) % 5000000 == 0:
        now = time.time()
        rate = 5000000 / (now - last_report) if (now - last_report) > 0 else 0
        print(f"Processed {i+1:,} reads ({rate:,.0f} reads/s) | ERR127306: {count6:,} | ERR127307: {count7:,} | Other: {count_other:,}")
        last_report = now

in_bam.close()
out6.close()
out7.close()

total_time = time.time() - start_time
print(f"=== Splitting Complete in {total_time:.1f}s ===")
print(f"Total ERR127306 reads: {count6:,}")
print(f"Total ERR127307 reads: {count7:,}")
print(f"Total Other reads: {count_other:,}")
