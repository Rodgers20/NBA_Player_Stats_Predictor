/**
 * Auto-focus the search input inside Dash 4 dcc.Dropdown when opened.
 *
 * Problem: Dash 4 single-select dropdown uses hidden radio buttons for a11y.
 * When the dropdown opens, Dash focuses the hidden radio for the selected option
 * instead of the visible search input. Users have to click the search box manually.
 *
 * Fix: After the dropdown opens, steal focus back to the search input.
 * Dash focuses the radio inside requestAnimationFrame, so we use a slightly
 * longer timeout (80ms) to run after Dash's rAF completes.
 */
(function () {
    'use strict';

    var DROPDOWN_ID = 'player-dropdown';

    function focusSearchInput() {
        var wrapper = document.querySelector('#' + DROPDOWN_ID + ' .dash-dropdown-wrapper, #' + DROPDOWN_ID);
        if (!wrapper) return;

        // The dropdown content is rendered as a Radix popover (portaled outside #player-dropdown)
        // Find it by data-state="open" or by class
        var content = document.querySelector('.dash-dropdown-content[data-state="open"]');
        if (!content) {
            // Fallback: find any open dropdown content
            content = document.querySelector('.dash-dropdown-content');
        }
        if (!content) return;

        var searchInput = content.querySelector('.dash-dropdown-search');
        if (searchInput && document.activeElement !== searchInput) {
            searchInput.focus();
        }
    }

    function attachToDropdown() {
        var trigger = document.querySelector('#' + DROPDOWN_ID + ' .dash-dropdown');
        if (!trigger || trigger.__autoFocusAttached) return;
        trigger.__autoFocusAttached = true;

        trigger.addEventListener('click', function () {
            // Dash focuses the radio option via requestAnimationFrame.
            // We wait ~80ms to ensure our focus runs AFTER Dash's rAF.
            setTimeout(focusSearchInput, 80);
        });

        // Also handle keyboard open (ArrowDown/Enter)
        trigger.addEventListener('keyup', function (e) {
            if (e.key === 'ArrowDown' || e.key === 'Enter') {
                setTimeout(focusSearchInput, 80);
            }
        });
    }

    // Watch for Dash to render the dropdown component
    var observer = new MutationObserver(function () {
        var trigger = document.querySelector('#' + DROPDOWN_ID + ' .dash-dropdown');
        if (trigger && !trigger.__autoFocusAttached) {
            attachToDropdown();
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });

    // Attempt immediately if already rendered
    if (document.readyState !== 'loading') {
        attachToDropdown();
    } else {
        document.addEventListener('DOMContentLoaded', attachToDropdown);
    }
})();
