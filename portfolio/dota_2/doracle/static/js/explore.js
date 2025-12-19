let heroNames = {};  // global map for id -> name

// Load JSON first
$.getJSON("{{ url_for('static', filename='data/heroes.json') }}", function(data){
    data.forEach(hero => {
        heroNames[hero.id.toString()] = hero.name; // string keys
    });

    // Now attach click handler inside the callback
    $(".hero-image").click(function () {
        let hero_id = $(this).data("heroid");
        let heroIdStr = hero_id.toString();

        let url = `/stats/${hero_id}`; 
        $.get(url, function (data) {

            // Hero title
            $("#heroTitle").text(heroNames[heroIdStr] + " Stats");

            // Other details (best teammate)
            let teammateIdStr = data.best_teammate.toString();
            let otherHtml = `
                <div>Pick rate: ${(data.pickrate*100).toFixed(1)}%</div>
                <div>Win rate: ${(data.winrate*100).toFixed(1)}%</div>
                <div>
                    Best teammate: 
                    <img src="static/img/avatar-sb/${data.best_teammate}.png" width="40" height="25">
                    ${heroNames[teammateIdStr]} — Synergy: ${data.best_teammate_synergy.toFixed(2)}
                </div>
            `;
            $("#otherDetails").html(otherHtml);

            // Best matchups
            let bestHtml = "";
            data.best_matchups.forEach(item => {
                let idStr = item[0].toString();
                let val = item[1];
                bestHtml += `
                    <div class="matchup-box text-center">
                        <div class="matchup-name">${heroNames[idStr]}</div>
                        <img src="static/img/avatar-sb/${idStr}.png" width="40" height="25">
                        <div class="matchup-value">${val.toFixed(3)}</div>
                    </div>
                `;
            });
            $("#bestMatchups").html(bestHtml);

            // Worst counters
            let worstHtml = "";
            data.worst_counters.forEach(item => {
                let idStr = item[0].toString();
                let val = item[1];
                worstHtml += `
                    <div class="matchup-box text-center">
                        <img src="static/img/avatar-sb/${idStr}.png" width="40" height="25">
                        <div class="matchup-name">${heroNames[idStr]}</div>
                        <div class="matchup-value">${val.toFixed(3)}</div>
                    </div>
                `;
            });
            $("#worstCounters").html(worstHtml);
        });
    });
});

