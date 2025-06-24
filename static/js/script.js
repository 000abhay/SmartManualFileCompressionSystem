document.addEventListener('DOMContentLoaded', function() {
    // SPA Navigation
    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.page-section');
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            navBtns.forEach(b => b.removeAttribute('aria-current'));
            btn.setAttribute('aria-current', 'page');
            sections.forEach(sec => sec.classList.remove('active'));
            document.getElementById(btn.dataset.page).classList.add('active');
            document.getElementById(btn.dataset.page).focus();
        });
    });

    // Manual compress/decompress
    document.getElementById('compress-btn').addEventListener('click', function() {
        const filename = document.getElementById('filename').value;
        const algorithm = document.getElementById('algorithm').value;
        const progressDiv = document.getElementById('manual-progress');
        progressDiv.textContent = 'Compressing...';
        fetch('/compress', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `filename=${encodeURIComponent(filename)}&algorithm=${encodeURIComponent(algorithm)}`
        }).then(response => response.json())
          .then(data => {
              progressDiv.textContent = data.message || data.error;
          }).catch(err => {
              progressDiv.textContent = 'Error: ' + err;
          });
    });

    document.getElementById('decompress-btn').addEventListener('click', function() {
        const filename = document.getElementById('filename').value;
        const algorithm = document.getElementById('algorithm').value;
        fetch('/decompress', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `filename=${encodeURIComponent(filename)}&algorithm=${encodeURIComponent(algorithm)}`
        }).then(response => response.json())
          .then(data => {
              alert(data.message || data.error);
          });
    });

    // Upload and compress
    document.getElementById('upload-btn').addEventListener('click', function() {
        const fileInput = document.getElementById('file-input');
        const algorithm = document.getElementById('upload-algorithm').value;
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('algorithm', algorithm);

        const progressDiv = document.getElementById('upload-progress');
        progressDiv.textContent = 'Compressing...';
        fetch('/upload', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
          .then(data => {
              progressDiv.textContent = data.message || data.error;
          }).catch(err => {
              progressDiv.textContent = 'Error: ' + err;
          });
    });

    // Activity Log: Pagination, Search, Filter
    const ACTIVITY_PAGE_SIZE = 10;
    let activityData = [];
    let filteredData = [];
    let currentPage = 1;

    function renderActivityList() {
        const list = document.getElementById('activity-list');
        list.innerHTML = '';
        const start = (currentPage - 1) * ACTIVITY_PAGE_SIZE;
        const end = start + ACTIVITY_PAGE_SIZE;
        const pageItems = filteredData.slice(start, end);
        if (pageItems.length === 0) {
            list.innerHTML = '<li>No activity found.</li>';
        } else {
            pageItems.forEach(entry => {
                let icon = '<span class="icon info">&#9679;</span>';
                if (/compressed/i.test(entry)) icon = '<span class="icon success">&#10003;</span>';
                if (/decompressed/i.test(entry)) icon = '<span class="icon success">&#8681;</span>';
                if (/fail|error/i.test(entry)) icon = '<span class="icon fail">&#10007;</span>';
                list.innerHTML += `<li>${icon} <span>${entry}</span></li>`;
            });
        }
        document.getElementById('page-info').textContent =
            `Page ${currentPage} of ${Math.max(1, Math.ceil(filteredData.length / ACTIVITY_PAGE_SIZE))}`;
    }

    function filterActivity() {
        const search = document.getElementById('activity-search').value.trim().toLowerCase();
        const filter = document.getElementById('activity-filter').value;
        filteredData = activityData.filter(line => {
            let match = true;
            if (search) match = line.toLowerCase().includes(search);
            if (filter === 'compressed') match = match && /compressed/i.test(line);
            if (filter === 'decompressed') match = match && /decompressed/i.test(line);
            if (filter === 'failed') match = match && (/fail|error/i.test(line));
            return match;
        });
        currentPage = 1;
        renderActivityList();
    }

    document.getElementById('activity-search').addEventListener('input', filterActivity);
    document.getElementById('activity-filter').addEventListener('change', filterActivity);

    document.getElementById('prev-page').addEventListener('click', function() {
        if (currentPage > 1) {
            currentPage--;
            renderActivityList();
        }
    });
    document.getElementById('next-page').addEventListener('click', function() {
        if (currentPage < Math.ceil(filteredData.length / ACTIVITY_PAGE_SIZE)) {
            currentPage++;
            renderActivityList();
        }
    });

    // Fetch activity log (assume static file for demo)
    fetch('/static/activity.log')
        .then(res => res.text())
        .then(text => {
            activityData = text.trim().split('\n').reverse();
            filteredData = activityData;
            renderActivityList();
        });

    // Accessibility: keyboard nav for nav bar
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('keydown', e => {
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                let next = btn.parentElement.nextElementSibling;
                if (next) next.querySelector('.nav-btn').focus();
            }
            if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                let prev = btn.parentElement.previousElementSibling;
                if (prev) prev.querySelector('.nav-btn').focus();
            }
        });
    });
});
